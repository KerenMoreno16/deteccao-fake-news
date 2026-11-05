# -*- coding: utf-8 -*-
"""
bilstm.py — Bidirectional LSTM text classifier
(PyTorch).
Correções aplicadas:
  - f1_score importado no topo do arquivo em vez de
    dentro de _evaluate_loader().
  - evaluate() usa inferência única via predict_proba()
    eliminando dupla chamada ao modelo.
  - evaluate() verifica _is_fitted antes de inferir.
  - fit(): aviso quando classe ausente em y_train.
  - _evaluate_loader(): labels=[0, 1] adicionado para
    consistência com bertimbau.py.
  - load(): fallback weights_only=False para
    PyTorch < 2.4.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from src.features.embeddings import (
    Vocabulary,
    collate_batch,
    load_fasttext_embeddings,
)
from torch.serialization import add_safe_globals
add_safe_globals([Vocabulary])
from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_len: int,
    ):
        self.encodings = [
            vocab.encode(t, max_len)
            for t in texts
        ]
        self.labels = labels
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[List[int], int]:
        enc = self.encodings[idx]
        return (
            enc if enc else [1],
            self.labels[idx],
        )
# ---------------------------------------------------------------------------
# BiLSTM architecture
# ---------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        embedding_matrix: Optional[
            torch.Tensor
        ] = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                embedding_matrix
            )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout if num_layers > 1 else 0.0
            ),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )
        return self.classifier(
            self.dropout(last_hidden)
        )
# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------
class BiLSTMModel:
    def __init__(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 128,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        use_pretrained_embeddings: bool = True,
        fasttext_path: Optional[str] = None,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.embedding_dim           = embedding_dim
        self.hidden_dim              = hidden_dim
        self.num_layers              = num_layers
        self.dropout                 = dropout
        self.max_seq_len             = max_seq_len
        self.batch_size              = batch_size
        self.learning_rate           = learning_rate
        self.max_epochs              = max_epochs
        self.early_stopping_patience = (
            early_stopping_patience
        )
        self.use_pretrained_embeddings = (
            use_pretrained_embeddings
        )
        self.fasttext_path  = fasttext_path
        self.random_state   = random_state
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.vocab:  Optional[Vocabulary]       = None
        self.model:  Optional[BiLSTMClassifier] = None
        self._is_fitted = False
        self.train_time: float = 0.0
        self.history: Dict[str, List] = {
            "epoch":      [],
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
        }
        if (
            self.use_pretrained_embeddings
            and not self.fasttext_path
        ):
            logger.warning(
                "BiLSTMModel: "
                "use_pretrained_embeddings=True "
                "mas fasttext_path não fornecido. "
                "Embeddings serão aleatórios."
            )
    # -----------------------------------------------------------------------
    # Prepare data
    # -----------------------------------------------------------------------
    def _prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = TextDataset(
            texts,
            labels,
            self.vocab,
            self.max_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
        )
    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BiLSTMModel":
        """
        Treina o modelo BiLSTM.
        Parâmetros
        ----------
        X_train : textos de treino.
        y_train : labels de treino.
        X_val   : textos de validação (opcional).
                  Se fornecido, habilita early
                  stopping por val_f1 e restauração
                  do melhor estado.
        y_val   : labels de validação.
        Retorna
        -------
        self
        """
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.random_state
            )
        self.history = {
            "epoch":      [],
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
        }
        # Vocabulário apenas com textos de treino
        self.vocab = Vocabulary(min_freq=2)
        self.vocab.build(X_train)
        # Embeddings
        embedding_matrix = None
        if (
            self.use_pretrained_embeddings
            and self.fasttext_path
        ):
            embedding_matrix = (
                load_fasttext_embeddings(
                    self.fasttext_path,
                    self.vocab,
                    self.embedding_dim,
                )
            )
            if embedding_matrix is not None:
                self.embedding_dim = (
                    embedding_matrix.shape[1]
                )
        # Modelo
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=2,
            dropout=self.dropout,
            embedding_matrix=embedding_matrix,
        ).to(self.device)
        # DataLoaders
        train_loader = self._prepare_data(
            X_train,
            y_train.tolist(),
            shuffle=True,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._prepare_data(
                X_val,
                y_val.tolist(),
            )
        # Optimizer e scheduler
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=2,
            factor=0.5,
        )
        # Ponderação de classes
        _counts = np.bincount(
            np.asarray(y_train, dtype=int),
            minlength=2,
        ).astype(float)
        _w = np.where(
            _counts > 0,
            _counts.sum() / (
                2.0 * np.maximum(_counts, 1.0)
            ),
            0.0,
        )
        # Aviso se classe ausente
        if np.any(_counts == 0):
            missing = [
                i for i, c in enumerate(_counts)
                if c == 0
            ]
            logger.warning(
                "BiLSTM: classe(s) %s ausente(s) "
                "em y_train. "
                "Peso definido como 0.0 — classe "
                "ignorada na loss. "
                "Verifique o dataset.",
                missing,
            )
        class_weights = torch.tensor(
            _w, dtype=torch.float, device=self.device
        )
        logger.info(
            "BiLSTM class weights (real, fake): %s",
            _w.round(4).tolist(),
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )
        best_val_f1      = 0.0
        patience_counter = 0
        best_state       = None
        start            = time.time()
        # ----------------------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------------------
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_loss = 0.0
            for (
                batch_x,
                batch_lengths,
                batch_y,
            ) in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(
                    batch_x, batch_lengths
                )
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()
                total_loss += loss.item()
            avg_loss = (
                total_loss / len(train_loader)
            )
            # ------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------
            if val_loader is not None:
                val_f1, val_loss = (
                    self._evaluate_loader(
                        val_loader, criterion
                    )
                )
                scheduler.step(val_loss)
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(
                    avg_loss
                )
                self.history["val_loss"].append(
                    val_loss
                )
                self.history["val_f1"].append(val_f1)
                logger.info(
                    "Epoch %02d | "
                    "train_loss=%.4f | "
                    "val_loss=%.4f | "
                    "val_f1=%.4f",
                    epoch, avg_loss,
                    val_loss, val_f1,
                )
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = {
                        k: v.clone()
                        for k, v in
                        self.model.state_dict()
                        .items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if (
                        patience_counter
                        >= self.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping "
                            "no epoch %d",
                            epoch,
                        )
                        break
            else:
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(                    avg_loss
                )
                self.history["val_loss"].append(
                    np.nan
                )
                self.history["val_f1"].append(
                    np.nan
                )
                logger.info(
                    "Epoch %02d | train_loss=%.4f",
                    epoch, avg_loss,
                )
        # ----------------------------------------------------------------
        # Restaurar melhor estado
        # ----------------------------------------------------------------
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                "Melhor estado restaurado "
                "(val_f1=%.4f)",
                best_val_f1,
            )
        self.train_time = time.time() - start
        self._is_fitted = True
        logger.info(
            "BiLSTM training time: %.2fs",
            self.train_time,
        )
        return self
    # -----------------------------------------------------------------------
    # Validation (interno — usado durante o treino)
    # -----------------------------------------------------------------------
    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Avalia o modelo em um DataLoader.
        f1_score importado no topo do arquivo.
        labels=[0, 1] garante cálculo correto
        mesmo quando uma classe está ausente
        nas predições do batch.
        Retorna
        -------
        (f1_macro, avg_loss)
        """
        self.model.eval()
        all_preds  = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for (
                batch_x,
                batch_lengths,
                batch_y,
            ) in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits  = self.model(
                    batch_x, batch_lengths
                )
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                preds = (
                    logits
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                all_preds.extend(preds.tolist())
                all_labels.extend(
                    batch_y.cpu().numpy().tolist()
                )
        # labels=[0, 1] — consistente com bertimbau.py
        f1 = f1_score(
            all_labels,
            all_preds,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        avg_loss = total_loss / len(loader)
        return f1, avg_loss
    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------
    def predict(
        self,
        X: List[str],
    ) -> np.ndarray:
        """
        Prediz classes para os textos fornecidos.
        Retorna argmax de predict_proba() para
        garantir consistência com as probabilidades.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling predict()."
            )
        return self.predict_proba(X).argmax(axis=1)
    def predict_proba(
        self,
        X: List[str],
    ) -> np.ndarray:
        """
        Retorna probabilidades por classe via
        softmax.
        Parâmetros
        ----------
        X : lista de textos.
        Retorna
        -------
        np.ndarray de shape (n_samples, 2).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling predict_proba()."
            )
        loader = self._prepare_data(
            X, [0] * len(X)
        )
        self.model.eval()
        probas = []
        with torch.no_grad():
            for batch_x, batch_lengths, _ in (
                loader
            ):
                batch_x = batch_x.to(self.device)
                logits  = self.model(
                    batch_x, batch_lengths
                )
                p = (
                    torch.softmax(logits, dim=1)
                    .cpu()
                    .numpy()
                )
                probas.append(p)
        return np.vstack(probas)
    # -----------------------------------------------------------------------
    # Evaluation (externo — chamado por train.py)
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste.
        Inferência executada uma única vez via
        predict_proba() para evitar dupla chamada
        ao modelo.
        Parâmetros
        ----------
        X_test      : textos de teste.
        y_test      : labels verdadeiros.
        label_names : nomes das classes.
        Retorna
        -------
        dict com métricas completas.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling evaluate()."
            )
        start   = time.time()
        y_proba = self.predict_proba(X_test)
        y_pred  = y_proba.argmax(axis=1)
        inference_time = time.time() - start
        metrics = compute_metrics(
            y_test,
            y_pred,
            y_proba,
            label_names,
        )
        metrics["train_time_seconds"]     = (
            self.train_time
        )
        metrics["inference_time_seconds"] = (
            inference_time
        )
        return metrics
    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Salva modelo BiLSTM no disco.
        Estrutura salva (arquivo único .pt):
            model_state : pesos do modelo
            vocab       : instância de Vocabulary
            config      : hiperparâmetros completos
        Nota: fasttext_path é salvo apenas para
        fins informativos. Os pesos dos embeddings
        já estão em model_state e NÃO são
        re-carregados do arquivo .bin no load().
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            {
                "model_state": (
                    self.model.state_dict()
                ),
                "vocab": self.vocab,
                "config": {
                    "embedding_dim": (
                        self.embedding_dim
                    ),
                    "hidden_dim":    self.hidden_dim,
                    "num_layers":    self.num_layers,
                    "dropout":       self.dropout,
                    "max_seq_len":   self.max_seq_len,
                    "batch_size":    self.batch_size,
                    "learning_rate": (
                        self.learning_rate
                    ),
                    "max_epochs":    self.max_epochs,
                    "early_stopping_patience": (
                        self.early_stopping_patience
                    ),
                    "use_pretrained_embeddings": (
                        self.use_pretrained_embeddings
                    ),
                    "fasttext_path": (
                        self.fasttext_path
                    ),
                    "random_state":  self.random_state,
                },
            },
            path,
        )
        logger.info(
            "BiLSTM model saved: %s", path
        )
    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> "BiLSTMModel":
        """
        Carrega modelo BiLSTM do disco.
        Parâmetros
        ----------
        path   : caminho para o arquivo .pt.
        device : dispositivo ('cpu' ou 'cuda').
        Notas
        -----
        - fasttext_path é restaurado do checkpoint
          para fins informativos apenas.
        - Os pesos dos embeddings são restaurados
          de model_state, não re-carregados do .bin.
        - Checkpoints antigos sem fasttext_path
          recebem use_pretrained_embeddings=False
          para suprimir WARNING.
        - weights_only=True requer PyTorch >= 2.4
          com add_safe_globals([Vocabulary]).
          Fallback automático para versões anteriores.
        """
        try:
            checkpoint = torch.load(
                path,
                map_location=device or "cpu",
                weights_only=True,
            )
        except Exception:
            logger.warning(
                "BiLSTMModel.load: "
                "weights_only=True falhou. "
                "Tentando weights_only=False. "
                "Atualize para PyTorch >= 2.4 "
                "para suporte completo a "
                "add_safe_globals()."
            )
            checkpoint = torch.load(
                path,
                map_location=device or "cpu",
                weights_only=False,
            )
        cfg           = checkpoint["config"]
        fasttext_path = cfg.get(
            "fasttext_path", None
        )
        use_pretrained = cfg.get(
            "use_pretrained_embeddings", True
        )
        # Checkpoints antigos sem fasttext_path:
        # embeddings já estão em model_state
        if use_pretrained and fasttext_path is None:
            use_pretrained = False
            logger.debug(
                "BiLSTMModel.load: checkpoint sem "
                "fasttext_path. "
                "use_pretrained_embeddings=False "
                "para suprimir WARNING. "
                "Pesos restaurados de model_state."
            )
        model_inst = cls(
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
            batch_size=cfg.get("batch_size", 64),
            learning_rate=cfg.get(
                "learning_rate", 0.001
            ),
            max_epochs=cfg.get(
                "max_epochs", 30
            ),
            early_stopping_patience=cfg.get(
                "early_stopping_patience", 5
            ),
            use_pretrained_embeddings=use_pretrained,
            fasttext_path=fasttext_path,
            random_state=cfg.get(
                "random_state", 42
            ),
            device=device,
        )
        model_inst.vocab = checkpoint["vocab"]
        model_inst.model = BiLSTMClassifier(
            vocab_size=len(model_inst.vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_classes=2,
            dropout=cfg["dropout"],
        ).to(model_inst.device)
        model_inst.model.load_state_dict(
            checkpoint["model_state"]
        )
        model_inst._is_fitted = True
        logger.info(
            "BiLSTM model loaded: %s (device=%s)",
            path, model_inst.device,
        )
        return model_inst