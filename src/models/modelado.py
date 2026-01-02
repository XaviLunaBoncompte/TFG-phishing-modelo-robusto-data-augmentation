import json
import pandas as pd
import numpy as np
import sys
import joblib

from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from datetime import datetime

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
    RUTA_DATASET,
    OUTPUT_DIR,
    RANDOM_STATE
)

from ..feat_aux.preprocesamiento import (
    preprocesar_dataset
)

from ..feat_aux.vectorizacion_tfidf import (
    dividir_train_test_estratificado,
    construir_tfidf,
    ajustar_y_vectorizar,
    calc_dispersion,
    guardar_meta
) 

VERSION = "v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cargar_preprocesar_dt () -> tuple[pd.DataFrame, pd.DataFrame]:
    #Se relaiza la carga del dataset unificado y se aplica la limpieza ligera junto con el preprocesado
    print(f"\nCargando dataset desde: {RUTA_DATASET}")
    dt_unificado = pd.read_csv(RUTA_DATASET)

    dt = preprocesar_dataset(dt_unificado)
    print("\nForma tras preprocesar:", dt.shape)

    dt_train, dt_test = dividir_train_test_estratificado(dt)

    print("\nTrain:", dt_train.shape)
    print("Test:", dt_test.shape)

    return dt_train, dt_test

def const_vec_tfidf(dt_train: pd.DataFrame, dt_test: pd.DataFrame, version: str):
    #Contruye y ajusta el TF-IDF

    tfidf = construir_tfidf(
    ngram_range=(1, 2),                #Utiliza unigramas (una sola palabra) y bigramas (2 palabras consecutivas) para capturar contexto
    min_dt=5,                           #Se elimina palabras que aparecen menos de 5 veces, ya que menos cantidad de 5 reudce el ruido
    max_dt=0.9,                         #Se ignora cuando aparecen mas del 90%
    max_features=50000                  #Se evita generar matrices gigantes
    )

    x_train, x_test, y_train, y_test = ajustar_y_vectorizar(tfidf, dt_train, dt_test)

    print("\nx_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("\nEtiquetas unicas:", np.unique(y_train))

    disp_train = calc_dispersion(x_train)
    disp_test = calc_dispersion(x_test)
    print(f"\nDispersion del train: {disp_train:.4f}")
    print(f"Dispersion del test: {disp_test:.4f}")

    guardar_meta(tfidf, dt_train, dt_test, OUTPUT_DIR, version=version)

    return tfidf, x_train, x_test, y_train, y_test, disp_train, disp_test

def entrenar_svm(x_train, y_train, metodo: str = "sigmoid", cv: int = 3):
    #Se entrena el modelo base SVM y se calibra para obtener probabilidades, pudiendo usar soft voting mas en adelante

    svm_base = LinearSVC(
    class_weight="balanced",            #Se compensa el dataset desbalanceado
    random_state=RANDOM_STATE
    )

    #Se calibra el svm base para obtener probabilidades mediandte el metodo sigmoid y 3 folds para validacion cruzada
    calibracion = CalibratedClassifierCV(estimator=svm_base, method=metodo, cv=cv)

    calibracion.fit(x_train, y_train)

    return calibracion

def const_pipeline_mlp() -> Pipeline:
    #Se crea el pipelina con el TruncatedSVD y MLP

    mlp_pipeline = Pipeline(
        steps=[
            ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),   #1ª capa con 128 neuronas y la 2ª con 64 neuronas siendo una arquitectura decreciente muy comun en clasificadores
                activation="relu",              #Funcion de activacion relu, siendo rapida y estandar
                solver="adam",                  #Es el optimizador adaptativo muy usado
                random_state=RANDOM_STATE,
                max_iter=40,                    #Maximo de iteraciones
                early_stopping=True,            #Detiene el entrenamiento si no mejora en 3 ocasiones seguidas
                n_iter_no_change=3,
                verbose=True                    #Se muestra el entrenamiento
                )
            )
        ]
    )

    return mlp_pipeline

def entrenar_mlp(x_train, y_train) -> Pipeline:
    #Se entrena el modelo complementario a traves del pipeline SVD + MLP
    mlp = const_pipeline_mlp()
    mlp.fit(x_train, y_train)

    return mlp

def evaluacion_modelos(nombre: str, modelo, x_test, y_test, exist_prob: bool = False):
    #Se calcula las metricas para el modelo pasado

    y_pred = modelo.predict(x_test)

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0) 
    }

    if exist_prob == True:
        y_prob = modelo.predict_proba(x_test)[:, 1]
        metricas["roc_auc"] = roc_auc_score(y_test, y_prob)

    print(f"\n.: {nombre} :.")
    print(f"Accuracy:   {metricas['accuracy']:.4f}")
    print(f"Precision:   {metricas['precision']:.4f}")
    print(f"Recall:   {metricas['recall']:.4f}")
    print(f"F1_macro:   {metricas['f1_macro']:.4f}")
    print(f"F1_weighted:   {metricas['f1_weighted']:.4f}")

    if exist_prob:
        print(f"ROC_AUC:   {metricas['roc_auc']:.4f}")
    else:
        print("No aplicable al ser modelo sin probabilidades")

    print(f"\nReport de clasificación {nombre}:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

    return metricas

def entrenar_ensamble_hv(version: str = VERSION):
    #Se ejecuta el pipeline completo:
        #Se cargan los datos
        #Se preprocesan
        #Se entrenan los modelos individuales y el Hard Voting
        #Se evaluan los modelos

    #Se identifica la ejecucion con un identificador unico
    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"\nRun ID: {run_id}")

    config = {
        "run_id": run_id,
        "version": version,
        "random_state": RANDOM_STATE,
        "vectorizacion": {
            "metodo": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 5,
            "max_df": 0.9,
            "max_features": 50000
        }
    }

    #1 Cargamos y preprocesamos el dataset
    dt_train, dt_test = cargar_preprocesar_dt()

    #Contruimos y ajsutamso el TF-IDF
    tfidf, x_train, x_test, y_train, y_test, disp_train, disp_test = (
        const_vec_tfidf(dt_train, dt_test, version)
    )

    svm = entrenar_svm(x_train, y_train, metodo="sigmoid", cv=3)
    mlp = entrenar_mlp(x_train, y_train)

    metricas_svm = evaluacion_modelos("SVM", svm, x_test, y_test, exist_prob=True)
    metricas_mlp = evaluacion_modelos("MLP (SVD)", mlp, x_test, y_test, exist_prob=True)

    #4 Se crea el modelo en ensamblado que combina ambos modelos (SVM y MLP)
    ens = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("mlp", mlp)
        ],
        voting="soft",
        weights=[2, 1]                      #Se prioriza SVM que es el que da mejores resultados, sobretodo para reducir FN.
    )

    ens.fit(x_train, y_train)

    metricas_ens = evaluacion_modelos("Ensamble (Soft Voting)", ens, x_test, y_test, exist_prob=True)

    end_time = datetime.now()

    #5 Se guardan los modelos y sus versiones
    svm_path = OUTPUT_DIR / f"modelo_svm_{version}.joblib"
    mlp_path = OUTPUT_DIR / f"modelo_mlp_{version}.joblib"
    ens_path = OUTPUT_DIR / f"modelo_ens_{version}.joblib"

    config["ejecucion"] = {
        "inicio": start_time.isoformat(timespec="seconds"),
        "fin": end_time.isoformat(timespec="seconds"),
        "duracion": int((end_time - start_time).total_seconds())
    }

    config["datos"] = {
        "train_size": int(len(dt_train)),
        "test_size": int(len(dt_test)),
        "x_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "x_test_shape": [int(x_test.shape[0]), int(x_test.shape[1])]
    }

    config["modelos"] = {
        "svm_path": str(svm_path),
        "mlp_path": str(mlp_path),
        "ens_path": str(ens_path)
    }

    config_path = OUTPUT_DIR / f"config_{version}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    joblib.dump(svm, svm_path)
    joblib.dump(mlp, mlp_path)
    joblib.dump(ens, ens_path)

    metricas = {
            "run_id": run_id,
            "version": version,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "random_state": RANDOM_STATE,
            "dispersion": {
                "train": float(disp_train),
                "test": float(disp_test)
            },
            "modelos": {
                "SVM": {**metricas_svm, "path": str(svm_path)},
                "MLP": {**metricas_mlp, "path": str(mlp_path)},
                "Ensamble": {**metricas_ens, "path": str(ens_path)}
            }
    }

    metricas_path = OUTPUT_DIR / f"metricas_modelos_{version}.json"
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)

    print("Modelos guardados:")
    print("SVM:", svm_path)
    print("MLP:", mlp_path)
    print("Ensemble:", ens_path)
    print("Metricas:", metricas_path)
    print("Config:", config_path)


if __name__ == "__main__":
    entrenar_ensamble_hv()

    