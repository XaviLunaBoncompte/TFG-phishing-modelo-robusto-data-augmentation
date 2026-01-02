import json
import random
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

#Se importan las funciones y rutas creadas en los modulos .py de /feat_aux y /config

from ..config.rutas import (
    OUTPUT_DIR,
    RANDOM_STATE
)

from ..feat_aux.variantes_evasivas import (
    generar_variantes_evasivas,
    fijar_semillas
) 

@dataclass
class EvasionConfig:
    nivel: str = "fuerte"
    prob_semantica: float = 0.10
    prob_template: float = 0.60
    prob_perturbacion: float = 0.80
    num_muestras_semantica: int = 50

def _cargar_test(version: str) -> pd.DataFrame:

    #Se carga el test guardado anteriormente.

    test_path = OUTPUT_DIR / f"test_{version}.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"No se encuentra {test_path}")
    
    dt_test = pd.read_csv(test_path)
    requerido = {"texto_preprocesado", "label", "vector"}

    falta = requerido - set(dt_test.columns)
    if falta:
        raise ValueError(f"El test {test_path} no tiene columnas requeridas")
    
    return dt_test

def _generar_test_evasivo(dt_test: pd.DataFrame, cfg: EvasionConfig, seed: int = RANDOM_STATE) -> pd.DataFrame:

    #Se genera la columna texto_evasivo para label=1 y mantiene texto_preprocesado en la columna texto_evasivo para label=0

    dt_test_evasivo = dt_test.copy()

    fijar_semillas(seed)

    dt_test_evasivo["texto_evasivo"] = dt_test_evasivo["texto_preprocesado"]

    mascara = dt_test_evasivo["label"] == 1          #Se seleccionan los textos maliciosos del test para apllicar las variantes evasivas
    dt_mal = dt_test_evasivo.loc[mascara].copy()

    semanticas = dt_mal.sample(n=min(cfg.num_muestras_semantica, len(dt_mal)), random_state=seed).index

    errores = 0

    def try_except_variantes_evasivas(r):
        
        nonlocal errores
        try:
            prob_sem = cfg.prob_semantica if r.name in semanticas else 0.0

            return generar_variantes_evasivas(               
                    r, 
                    nivel=cfg.nivel,
                    prob_semantica=prob_sem,                            #Se escoge una probabilidad baja para la semantica para no saturar la CPU con el back_translation
                    prob_template=cfg.prob_template,
                    prob_perturbacion=cfg.prob_perturbacion
            )             
        except Exception:
            errores += 1
            return r["texto_preprocesado"]
    
    dt_test_evasivo.loc[mascara, "texto_evasivo"] = dt_test_evasivo.loc[mascara].apply(try_except_variantes_evasivas, axis=1)

    orig = dt_test.loc[mascara, "texto_preprocesado"]
    eva = dt_test_evasivo.loc[mascara, "texto_evasivo"]

    ratio_de_cambio = float((eva != orig).mean()) if len(orig) else 0.0
    print(f"\nPorcentaje de phishing modificados por evasion: {ratio_de_cambio*100:.2f}%")
    print(f"Excepciones producidas durante la evasion: {errores}")

    ejemplos_path = OUTPUT_DIR / "ejemplos_test_evasivo.csv"
    dt_test_evasivo.loc[mascara, ["vector", "texto_preprocesado", "texto_evasivo"]].head(200).to_csv(
        ejemplos_path, index=False
    )

    print(f"\nEjemplos guardados en: {ejemplos_path}")

    dt_test_evasivo.attrs["ratio_de_cambios_evasion"] = ratio_de_cambio
    dt_test_evasivo.attrs["ecxcepciones_evasion"] = errores

    return dt_test_evasivo

def _vectorizar(tfidf, textos: pd.Series):

    return tfidf.transform(textos)

def _evaluar(nombre: str, modelo, x, y, exist_prob: bool) -> Dict[str, Any]:

    y_pred = modelo.predict(x)

    salida: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0, output_dict=True)
    }

    if exist_prob:
        y_prob = modelo.predict_proba(x)[:, 1]
        salida["roc_auc"] = float(roc_auc_score(y, y_prob))

    print(f"\n.: {nombre} :.")
    print(f"Accuracy:   {salida['accuracy']:.4f}")
    print(f"Precision:   {salida['precision']:.4f}")
    print(f"Recall:   {salida['recall']:.4f}")
    print(f"F1_macro:   {salida['f1_macro']:.4f}")
    print(f"F1_weighted:   {salida['f1_weighted']:.4f}")

    if exist_prob:
        print(f"ROC_AUC:   {salida['roc_auc']:.4f}")
    else:
        print("No aplicable al ser modelo sin probabilidades")

    print("\nMatriz de confusion:")
    print(salida['confusion_matrix'])

    return salida

def _cargar_modelos(version: str) -> Tuple[Any, Any, Any, Any]:

    #Se carga tfidf y los modelos gaurdados anteriormente

    tfidf_path = OUTPUT_DIR / f"tfidf_vect_{version}.joblib"
    svm_path = OUTPUT_DIR / f"modelo_svm_{version}.joblib"
    mlp_path = OUTPUT_DIR / f"modelo_mlp_{version}.joblib"
    ens_path = OUTPUT_DIR / f"modelo_ens_{version}.joblib"

    for p in [tfidf_path, svm_path, mlp_path, ens_path]:
        if not p.exists():
            raise FileNotFoundError(f"No se encuentra {p}")
        
    tfidf = joblib.load(tfidf_path)
    svm = joblib.load(svm_path)
    mlp = joblib.load(mlp_path)
    ens = joblib.load(ens_path)

    return tfidf, svm, mlp, ens

def ejecutar_eval_rob(
    version_base: str = "v1",
    version_robust: str = "v2_robusto",
    evasion_cfg: EvasionConfig = EvasionConfig(),
    seed: int = RANDOM_STATE
) -> Path:
    
    #Se evalua las siguientes situaciones
    #   - Base (v1): test normal y test evasivo
    #   - Robusto (v2_robusto): test normal y test evasivo

    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"\nRun ID: {run_id}\n")

    #Se carga el test base
    dt_test = _cargar_test(version_base)

    #Se genera evasion a traves del test base
    dt_test_evasivo = _generar_test_evasivo(dt_test, evasion_cfg, seed)

    #Cargamos modelos y vectorizadores por cada version (base y robusta)
    tfidf_base, svm_base, mlp_base, ens_base = _cargar_modelos(version_base)
    tfidf_rob, svm_rob, mlp_rob, ens_rob = _cargar_modelos(version_robust)

    #Vectorizamos y evaluamos
    y = dt_test["label"].values

    x_base_norm = _vectorizar(tfidf_base, dt_test["texto_preprocesado"])
    x_base_eva = _vectorizar(tfidf_base, dt_test_evasivo["texto_evasivo"])

    x_rob_norm = _vectorizar(tfidf_rob, dt_test["texto_preprocesado"])
    x_rob_eva = _vectorizar(tfidf_rob, dt_test_evasivo["texto_evasivo"])

    resultados = {
        "run_id": run_id,
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "version_base": version_base,
        "version_robusta": version_robust,
        "config_evasion": asdict(evasion_cfg),
        "evaluaciones": {}
    }

    #Evaluaciones
    resultados["evaluaciones"]["base_test_normal"] = {
        "SVM": _evaluar("SVM Base (Normal)", svm_base, x_base_norm, y, True),
        "MLP": _evaluar("MLP Base (Normal)", mlp_base, x_base_norm, y, True),
        "Ensamble": _evaluar("Ensamble Base Soft (Normal)", ens_base, x_base_norm, y, True)
    }

    resultados["evaluaciones"]["base_test_evasivo"] = {
        "SVM": _evaluar("SVM Base (Evasivo)", svm_base, x_base_eva, y, True),
        "MLP": _evaluar("MLP Base (Evasivo)", mlp_base, x_base_eva, y, True),
        "Ensamble": _evaluar("Ensamble Base Soft (Evasivo)", ens_base, x_base_eva, y, True)
    }

    resultados["evaluaciones"]["robusto_test_normal"] = {
        "SVM": _evaluar("SVM robusto (Normal)", svm_rob, x_rob_norm, y, True),
        "MLP": _evaluar("MLP robusto (Normal)", mlp_rob, x_rob_norm, y, True),
        "Ensamble": _evaluar("Ensamble robusto Soft (Normal)", ens_rob, x_rob_norm, y, True)
    }

    resultados["evaluaciones"]["robusto_test_evasivo"] = {
        "SVM": _evaluar("SVM robusto (Evasivo)", svm_rob, x_rob_eva, y, True),
        "MLP": _evaluar("MLP robusto (Evasivo)", mlp_rob, x_rob_eva, y, True),
        "Ensamble": _evaluar("Ensamble robusto Soft (Evasivo)", ens_rob, x_rob_eva, y, True)
    }

    out_path = OUTPUT_DIR / f"metricas_robustez_{run_id}_{version_base}_vs_{version_robust}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print(f"\nMetricas de robustez guardadas.")
    return out_path

if __name__ == "__main__":
    ejecutar_eval_rob()