"""
bertimbau.py — Fine-tuning wrapper for BERTimbau and
BERT-base-uncased.
Uses Hugging Face Transformers and a standard PyTorch
training loop with mixed-precision support
(if CUDA available).
Correções aplicadas:
  - evaluate() usa inferência única via predict_proba()
    eliminando dupla chamada ao modelo.
  - evaluate() verifica _is_fitted antes de inferir.
  - Documentação de SUPPORTED_MODELS como restritivo
    por design.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BERTDataset(Dataset):
    """
    Dataset para fine-tuning de modelos transformer.
    Tokeniza e padda todos os textos em __init__
    para evitar re-tokenização por epoch.
    Nota: para datasets muito grandes (>100k textos),
    considerar tokenização lazy para evitar OOM.
    Os seis datasets da monografia estão dentro do
    limite seguro.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            labels, dtype=torch.long
        )
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        item = {
            k: v[idx]
            for k, v in self.encodings.items()
        }
        item["labels"] = self.labels[idx]
        return item
# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------
class TransformerClassifier:
    """
    Fine-tunes um modelo Transformer para classificação
    binária de texto.
    Modelos suportados:
        PT-BR   : neuralmind/bert-base-portuguese-cased
        English : bert-base-uncased
    SUPPORTED_MODELS é restritivo por design — apenas
    os dois modelos validados para a monografia são
    aceitos. Para adicionar novos modelos, atualizar
    SUPPORTED_MODELS e BERT_MODEL_BY_LANGUAGE em
    src/utils/constants.py.
    """
    SUPPORTED_MODELS = {
        "neuralmind/bert-base-portuguese-cased",
        "bert-base-uncased",
    }
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        epochs: int = 5,
        dropout: float = 0.1,
        gradient_clip: float = 1.0,
        fp16: bool = False,
        random_state: int = 42,
        device: Optional[str] = None,
        num_labels: int = 2,
    ):
        if model_name is None:
            raise ValueError(
                "model_name deve ser informado "
                "explicitamente. "
                "Use "
                "'neuralmind/bert-base-portuguese-cased'"
                " para PT-BR ou "
                "'bert-base-uncased' para inglês."
            )
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo não suportado: {model_name}. "
                f"Modelos permitidos: "
                f"{sorted(self.SUPPORTED_MODELS)}"
            )
        self.model_name    = model_name
        self.max_length    = max_length
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.warmup_ratio  = warmup_ratio
        self.epochs        = epochs
        self.dropout       = dropout
        self.gradient_clip = gradient_clip
        self.random_state  = random_state
        self.num_labels    = num_labels
        # Aviso se fp16 solicitado sem CUDA
        if fp16 and not torch.cuda.is_available():
            logger.warning(
                "fp16=True solicitado mas CUDA não "
                "disponível. "
                "Treinamento prosseguirá em fp32."
            )
        self.fp16 = fp16 and torch.cuda.is_available()
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.tokenizer  = None
        self.model      = None
        self._is_fitted = False
        self.train_time = 0.0
        self.history: Dict[str, List] = {
            "epoch":      [],
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
        }
    # -----------------------------------------------------------------------
    # Base model loader
    # -----------------------------------------------------------------------
    def _load_base_model(self) -> None:
        logger.info(
            "Carregando tokenizer e modelo: %s",
            self.model_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                hidden_dropout_prob=self.dropout,
                attention_probs_dropout_prob=(
                    self.dropout
                ),
            )
            .to(self.device)
        )
    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit(
        self,
        X_train: List[str],
        y_train,
        X_val: Optional[List[str]] = None,
        y_val=None,
    ) -> "TransformerClassifier":
        """
        Fine-tuna o modelo transformer.
        Parâmetros
        ----------
        X_train : textos de treino.
        y_train : labels de treino.
        X_val   : textos de validação (opcional).
                  Se fornecido, habilita early stopping
                  por val_f1 e restauração do melhor
                  estado.
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
        # Converter para list puro
        if hasattr(y_train, "tolist"):
            y_train = y_train.tolist()
        else:
            y_train = list(y_train)
        if y_val is not None:
            if hasattr(y_val, "tolist"):
                y_val = y_val.tolist()
            else:
                y_val = list(y_val)
        self.history = {
            "epoch":      [],
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
        }
        self._load_base_model()
        train_dataset = BERTDataset(
            X_train, y_train,
            self.tokenizer, self.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = BERTDataset(
                X_val, y_val,
                self.tokenizer, self.max_length,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=0,
            )
        total_steps  = len(train_loader) * self.epochs
        warmup_steps = int(
            total_steps * self.warmup_ratio
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = (
            torch.amp.GradScaler("cuda")
            if self.fp16
            else None
        )
        # Perda ponderada por classe (critério "balanced": n / (K * n_c)),
        # igual ao class_weight do sklearn e ao BiLSTM.
        _counts = np.bincount(
            np.asarray(y_train, dtype=int), minlength=2
        ).astype(float)
        _w = np.where(
            _counts > 0,
            _counts.sum() / (2.0 * np.maximum(_counts, 1.0)),
            0.0,
        )
        if np.any(_counts == 0):
            missing = [
                i for i, c in enumerate(_counts)
                if c == 0
            ]
            logger.warning(
                "BERT: classe(s) %s ausente(s) em y_train. "
                "Peso definido como 0.0 — classe ignorada "
                "na loss. Verifique o dataset.",
                missing,
            )
        self._class_weights = torch.tensor(
            _w, dtype=torch.float, device=self.device
        )
        logger.info("BERT class weights (real, fake): %s", _w.round(4).tolist())
        best_val_f1 = -np.inf
        best_state  = None
        start       = time.time()
        # ------------------------------------------------------------------
        # Epoch loop
        # ------------------------------------------------------------------
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
                labels = batch.pop("labels")
                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch)
                        loss = self._weighted_loss(
                            outputs.logits, labels
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(**batch)
                    loss = self._weighted_loss(
                        outputs.logits, labels
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            # ----------------------------------------------------------------
            # Validation
            # ----------------------------------------------------------------
            if val_loader is not None:
                val_f1, val_loss = (
                    self._evaluate_loader(val_loader)
                )
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(
                    avg_loss
                )
                self.history["val_loss"].append(
                    val_loss
                )
                self.history["val_f1"].append(val_f1)
                logger.info(
                    "Epoch %02d | train_loss=%.4f | "
                    "val_loss=%.4f | val_f1=%.4f",
                    epoch, avg_loss,
                    val_loss, val_f1,
                )
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    # Cópia em CPU: evita duplicar os pesos na memória da GPU
                    best_state  = {
                        k: v.detach().cpu().clone()
                        for k, v in
                        self.model.state_dict().items()
                    }
            else:
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(
                    avg_loss
                )
                self.history["val_loss"].append(np.nan)
                self.history["val_f1"].append(np.nan)
                logger.info(
                    "Epoch %02d | train_loss=%.4f",
                    epoch, avg_loss,
                )
        # ------------------------------------------------------------------
        # Restaurar melhor estado
        # ------------------------------------------------------------------
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                "Melhor estado restaurado "
                "(val_f1=%.4f)",
                best_val_f1,
            )
        else:
            logger.info(
                "Sem validação — usando modelo "
                "do último epoch."
            )
        self.train_time = time.time() - start
        self._is_fitted = True
        logger.info(
            "BERT training time: %.2fs",
            self.train_time,
        )
        return self
    # -----------------------------------------------------------------------
    # Validation (interno — usado durante o treino)
    # -----------------------------------------------------------------------
    def _weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy com pesos de classe (None → sem pesos)."""
        weight = getattr(self, "_class_weights", None)
        return nn.functional.cross_entropy(
            logits.float(), labels, weight=weight
        )

    def _evaluate_loader(
        self,
        loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Avalia o modelo em um DataLoader.
        Retorna
        -------
        (f1_macro, avg_loss)
        """
        self.model.eval()
        all_preds  = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
                labels = batch.pop("labels")
                outputs = self.model(**batch)
                total_loss += self._weighted_loss(
                    outputs.logits, labels
                ).item()
                preds = (
                    outputs.logits
                    .argmax(dim=-1)
                    .cpu()
                    .numpy()
                )
                all_preds.extend(preds.tolist())
                all_labels.extend(
                    labels
                    .cpu()
                    .numpy()
                    .tolist()
                )
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
        Retorna probabilidades por classe via softmax.
        Parâmetros
        ----------
        X : lista de textos.
        Retorna
        -------
        np.ndarray de shape (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling predict_proba()."
            )
        dataset = BERTDataset(
            X, [0] * len(X),
            self.tokenizer, self.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )
        self.model.eval()
        probas = []
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)
                p = (
                    torch.softmax(
                        outputs.logits, dim=-1
                    )
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
        CORRIGIDO: inferência executada uma única vez
        via predict_proba() para evitar dupla chamada
        ao modelo (~34s desnecessários por chamada
        no BERT).
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
        start = time.time()
        # CORRIGIDO: inferência única
        y_proba = self.predict_proba(X_test)
        y_pred  = y_proba.argmax(axis=1)
        inference_time = time.time() - start
        metrics = compute_metrics(
            y_test, y_pred, y_proba, label_names
        )
        metrics["train_time_seconds"]     = (
            self.train_time
        )
        metrics["inference_time_seconds"] = (
            inference_time
        )
        metrics["n_params"] = sum(
            p.numel()
            for p in self.model.parameters()
        )
        return metrics
    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Salva modelo, tokenizer e metadados em disco.
        Estrutura salva:
            path/
                config.json          (HuggingFace)
                pytorch_model.bin    (HuggingFace)
                tokenizer_config.json
                vocab.txt
                classifier_meta.json (hiperparâmetros)
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        meta = {
            "model_name":    self.model_name,
            "num_labels":    self.num_labels,
            "max_length":    self.max_length,
            "batch_size":    self.batch_size,
            "random_state":  self.random_state,
            "epochs":        self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay":  self.weight_decay,
            "warmup_ratio":  self.warmup_ratio,
        }
        with open(
            out / "classifier_meta.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=2)
        logger.info(
            "BERTimbau model saved: %s", path
        )
    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "TransformerClassifier":
        """
        Carrega modelo salvo do disco.
        Parâmetros
        ----------
        path       : diretório do modelo salvo.
        device     : dispositivo ('cpu' ou 'cuda').
                     Se None, detecta automaticamente.
        model_name : sobrescreve o model_name do meta.
                     Usar apenas quando classifier_meta.json
                     estiver ausente (checkpoints antigos).
        Raises
        ------
        ValueError
            Se model_name não puder ser determinado.
        """
        meta_path = Path(path) / "classifier_meta.json"
        saved_meta: Dict[str, Any] = {}
        if model_name is None:
            if meta_path.exists():
                with open(
                    meta_path, "r", encoding="utf-8"
                ) as f:
                    saved_meta = json.load(f)
                model_name = saved_meta.get(
                    "model_name"
                )
                if model_name is None:
                    raise ValueError(
                        "classifier_meta.json encontrado "
                        "mas 'model_name' ausente. "
                        "Salve o modelo novamente com "
                        "a versão corrigida de save()."
                    )
            else:
                raise ValueError(
                    "model_name não encontrado. "
                    "Informe explicitamente ou salve "
                    "o modelo com save() que persiste "
                    "classifier_meta.json."
                )
        else:
            if meta_path.exists():
                with open(
                    meta_path, "r", encoding="utf-8"
                ) as f:
                    saved_meta = json.load(f)
        inst = cls(
            model_name=model_name,
            num_labels=saved_meta.get(
                "num_labels", 2
            ),
            max_length=saved_meta.get(
                "max_length", 128
            ),
            batch_size=saved_meta.get(
                "batch_size", 32
            ),
            random_state=saved_meta.get(
                "random_state", 42
            ),
            epochs=saved_meta.get("epochs", 5),
            learning_rate=saved_meta.get(
                "learning_rate", 3e-5
            ),
            weight_decay=saved_meta.get(
                "weight_decay", 0.01
            ),
            warmup_ratio=saved_meta.get(
                "warmup_ratio", 0
             ),
            device=device,
        )
        inst.tokenizer = (
            AutoTokenizer.from_pretrained(path)
        )
        inst.model = (
            AutoModelForSequenceClassification
            .from_pretrained(path)
            .to(inst.device)
        )
        inst._is_fitted = True
        logger.info(
            "BERTimbau model loaded: %s (device=%s)",
            path, inst.device,
        )
        return inst