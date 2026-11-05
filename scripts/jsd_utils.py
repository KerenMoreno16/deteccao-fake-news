# -*- coding: utf-8 -*-
"""
scripts/jsd_utils.py
Utilitários para cálculo da divergência Jensen-Shannon (JSD)
entre distribuições empíricas de classes em treino e teste.
Utilizado pelo Experimento D (análise temporal) para quantificar
a distância entre as distribuições de classe do split de treino
e do split de teste.
Uso típico no pipeline EXP_D:
    from scripts.jsd_utils import (
        calcular_jsd_classes,
        registrar_jsd_no_json,
    )
    resultado = calcular_jsd_classes(y_train, y_test)
    metrics   = registrar_jsd_no_json(resultado, metrics)
    # salvar metrics no JSON do experimento
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Any, Dict, List, Optional, Union
def calcular_jsd_classes(
    y_train: Union[List, np.ndarray],
    y_test:  Union[List, np.ndarray],
    classes: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Calcula a divergência Jensen-Shannon (JSD) entre as distribuições
    empíricas de classes em treino e teste.
    Parâmetros
    ----------
    y_train : array-like
        Rótulos do conjunto de treino.
    y_test : array-like
        Rótulos do conjunto de teste.
    classes : list, opcional
        Classes a considerar. Padrão: [0, 1].
    Retorna
    -------
    dict com:
        train_distribution : list[float]
            Proporção de cada classe no treino.
        test_distribution : list[float]
            Proporção de cada classe no teste.
        jsd_bits : float
            Divergência JSD em bits (base 2).
            Intervalo: [0.0, 1.0].
            0.0 = distribuições idênticas.
            1.0 = distribuições completamente disjuntas.
        js_distance : float
            Distância JS = sqrt(JSD).
            Intervalo: [0.0, 1.0].
    Notas matemáticas
    -----------------
    scipy.spatial.distance.jensenshannon() retorna a DISTÂNCIA JS,
    definida como sqrt(JSD(P || Q)). Quando base=2 é passado
    explicitamente, o logaritmo utilizado é log₂ (não logaritmo
    natural), portanto o resultado está em bits (não nats).
    Para obter a divergência JSD a partir da distância:
        JSD = distancia ** 2
    Referência:
        Lin, J. (1991). Divergence measures based on the Shannon
        entropy. IEEE Transactions on Information Theory, 37(1),
        145-151.
    Raises
    ------
    ValueError
        Se y_train ou y_test estiverem vazios, ou se nenhuma
        amostra pertencer às classes especificadas.
    """
    if classes is None:
        classes = [0, 1]
    y_train_arr = np.asarray(y_train)
    y_test_arr  = np.asarray(y_test)
    if len(y_train_arr) == 0:
        raise ValueError(
            "y_train não pode ser vazio."
        )
    if len(y_test_arr) == 0:
        raise ValueError(
            "y_test não pode ser vazio."
        )
    # Distribuições empíricas
    p = np.array(
        [np.mean(y_train_arr == c) for c in classes],
        dtype=float,
    )
    q = np.array(
        [np.mean(y_test_arr == c) for c in classes],
        dtype=float,
    )
    # Normalização defensiva
    # Necessária quando classes não cobre todas as ocorrências
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0:
        raise ValueError(
            f"Nenhuma amostra de y_train pertence às classes "
            f"{classes}. Verifique os rótulos."
        )
    if q_sum == 0:
        raise ValueError(
            f"Nenhuma amostra de y_test pertence às classes "
            f"{classes}. Verifique os rótulos."
        )
    p = p / p_sum
    q = q / q_sum
    # jensenshannon com base=2 retorna distância JS em bits
    distancia   = float(jensenshannon(p, q, base=2))
    divergencia = distancia ** 2  # JSD em bits
    return {
        "train_distribution": p.tolist(),
        "test_distribution":  q.tolist(),
        "jsd_bits":           round(divergencia, 6),
        "js_distance":        round(distancia, 6),
    }
def registrar_jsd_no_json(
    resultado_jsd: Dict[str, Any],
    metrics_dict:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Insere o resultado de calcular_jsd_classes() no dicionário
    de métricas que será salvo como JSON pelo pipeline.
    Usa a chave padronizada 'js_divergence_unigrams' esperada por:
        - build_summary_csvs.py (build_temporal_summary)
        - generate_tables.py (gen_temporal)
        - generate_report.py (gen_exp_D_temporal)
    Parâmetros
    ----------
    resultado_jsd : dict
        Retorno de calcular_jsd_classes().
    metrics_dict : dict
        Dicionário de métricas do experimento (modificado in-place).
    Retorna
    -------
    O mesmo dicionário metrics_dict com as chaves adicionadas.
    Chaves adicionadas
    ------------------
    js_divergence_unigrams : float
        Divergência JSD em bits (base 2). Esta é a chave
        utilizada em todo o pipeline para consistência.
    js_distance : float
        Distância JS (raiz quadrada da divergência).
    train_class_dist : list[float]
        Distribuição empírica de classes no treino.
    test_class_dist : list[float]
        Distribuição empírica de classes no teste.
    Exemplo de uso no pipeline EXP_D
    ---------------------------------
    # Após realizar o split temporal e obter y_train, y_test:
    from scripts.jsd_utils import (
        calcular_jsd_classes,
        registrar_jsd_no_json,
    )
    jsd_result = calcular_jsd_classes(y_train, y_test)
    metrics    = registrar_jsd_no_json(jsd_result, metrics)
    # Salvar metrics no JSON:
    result_json = {
        "experiment_id": exp_id,
        "model":         model_name,
        "dataset":       dataset_name,
        "metrics":       metrics,
    }
    Path(output_path).write_text(
        json.dumps(result_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    """
    metrics_dict["js_divergence_unigrams"] = (
        resultado_jsd["jsd_bits"]
    )
    metrics_dict["js_distance"]      = (
        resultado_jsd["js_distance"]
    )
    metrics_dict["train_class_dist"] = (
        resultado_jsd["train_distribution"]
    )
    metrics_dict["test_class_dist"]  = (
        resultado_jsd["test_distribution"]
    )
    return metrics_dict
def verificar_balanceamento(
    resultado_jsd: Dict[str, Any],
    threshold_jsd: float = 0.05,
) -> Dict[str, Any]:
    """
    Verifica se o desbalanceamento entre treino e teste
    é aceitável.
    Parâmetros
    ----------
    resultado_jsd : dict
        Retorno de calcular_jsd_classes().
    threshold_jsd : float
        Limiar de JSD acima do qual o desbalanceamento é
        reportado como problemático. Padrão: 0.05 bits.
        Não existe consenso na literatura sobre um limiar
        universal; este valor é conservador e deve ser
        justificado na metodologia.
    Retorna
    -------
    dict com:
        balanced : bool
            True se JSD <= threshold_jsd.
        jsd_bits : float
            Valor da divergência.
        threshold : float
            Limiar utilizado.
        warning : str ou None
            Mensagem de aviso se desbalanceado.
    Raises
    ------
    ValueError
        Se resultado_jsd não contiver a chave 'jsd_bits'.
    """
    if "jsd_bits" not in resultado_jsd:
        raise ValueError(
            "resultado_jsd deve conter a chave 'jsd_bits'. "
            "Verifique se calcular_jsd_classes() foi chamado "
            "antes de verificar_balanceamento()."
        )
    jsd  = float(resultado_jsd["jsd_bits"])
    ok   = jsd <= threshold_jsd
    warn: Optional[str] = None
    if not ok:
        p = resultado_jsd.get(
            "train_distribution", []
        )
        q = resultado_jsd.get(
            "test_distribution", []
        )
        warn = (
            f"Desbalanceamento detectado entre treino "
            f"e teste: "
            f"JSD = {jsd:.4f} bits > limiar "
            f"{threshold_jsd:.4f} bits. "
            f"Treino: {[f'{v:.3f}' for v in p]}, "
            f"Teste: {[f'{v:.3f}' for v in q]}. "
            f"Considere estratificação ou reportar "
            f"como limitação."
        )
        print(f"[AVISO] {warn}")
    return {
        "balanced":  ok,
        "jsd_bits":  jsd,
        "threshold": threshold_jsd,
        "warning":   warn,
    }