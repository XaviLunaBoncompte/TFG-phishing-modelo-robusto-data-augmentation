import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

#RUTAS
from ..config.rutas import (
    RANDOM_STATE,
    TEST_SIZE
)


def dividir_train_test_estratificado(dt: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):

    #Divide el dataset en train y test estratificando por (vector, label)

    dt = dt.copy()

    dt["estrato"] = dt["vector"].astype(str) + "_" + dt["label"].astype(str)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    for train_idx, test_idx in sss.split(dt, dt["estrato"]):
        dt_train = dt.iloc[train_idx].reset_index(drop=True)
        dt_test = dt.iloc[test_idx].reset_index(drop=True)

    dt_train = dt_train.drop(columns=["estrato"])
    dt_test = dt_test.drop(columns=["estrato"])

    return dt_train, dt_test

def construir_tfidf(ngram_range=(1, 2), min_dt: int = 5, max_dt: float = 0.9, max_features: int = 50000) -> TfidfVectorizer:

    #Devuelve el vectorizador TF-IDF configurado con los parametros pasados

    return TfidfVectorizer(
    ngram_range=ngram_range,         #Se usa unigramas y bigramas
    min_df=min_dt,                   #Se elimina palabras que aparecen menos de 5 veces, ya que menos cantidad de 5 reudce el ruido
    max_df=max_dt,                 #Se ignora cuando aparecen mas del 90%
    max_features=max_features,         #Se evita generar matrices gigantes
    )

def ajustar_y_vectorizar(tfidf: TfidfVectorizer, dt_train: pd.DataFrame, dt_test: pd.DataFrame):

    #Ajusta el vectorizador con el texto de entrenamiento y transforma ambos splits en matrices numericas

    if "texto_preprocesado" not in dt_train.columns or "texto_preprocesado" not in dt_test.columns:
        raise ValueError("Debe existir la columna texto_preprocesado en ambos dt.")
    x_train_text = dt_train["texto_preprocesado"].tolist()
    x_test_text = dt_test["texto_preprocesado"].tolist()

    #Ajusta solo el conjunto de entrenamiento
    tfidf.fit(x_train_text)

    #Matriz de caracterisitcas
    x_train = tfidf.transform(x_train_text)
    x_test = tfidf.transform(x_test_text)

    #Se extrae las etiquetas como numpy (vector de etiquetas)
    y_train = dt_train["label"].to_numpy()
    y_test = dt_test["label"].to_numpy()

    return x_train, x_test, y_train, y_test

def calc_dispersion(matriz) -> float:

    #Calcula la dispersion, es decir, el porcentaje de ceros de un matriz dispersa.

    disp_m = 1.0 - (matriz.count_nonzero() / float(matriz.shape[0] * matriz.shape[1]))

    return disp_m 

def guardar_meta(tfidf, dt_train, dt_test, output_dir: Path, version: str = "v1"):

    #Se guarda el vectorizador TF-IDF entrenado, el csv de train y test, los metadatos con trazabilidad en un JSON

    output_dir.mkdir(parents=True, exist_ok=True)

    vect_path = output_dir / f"tfidf_vect_{version}.joblib"
    joblib.dump(tfidf, vect_path)

    train_csv = output_dir / f"train_{version}.csv"
    test_csv = output_dir / f"test_{version}.csv"

    col_min = ["id", "vector", "label", "texto_preprocesado"]

    dt_train[col_min].to_csv(train_csv, index=False)
    dt_test[col_min].to_csv(test_csv, index=False)

    #Se convierten los parametros a str para que sean serializables
    tfidf_params_orig = tfidf.get_params()
    tfidf_params = {k: str(v) for k, v in tfidf_params_orig.items()}

    #Se prepara los metadatos para la trazabilidad con un JSON
    m = {
        "version": version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "tfidf_params": tfidf_params,
        "train_size": len(dt_train),
        "test_size": len(dt_test),
        "class_dist_train": dt_train["label"].value_counts().to_dict(),
        "class_dist_test": dt_test["label"].value_counts().to_dict(),
        "vect_dist_train": dt_train["vector"].value_counts().to_dict(),
        "vect_dist_test": dt_test["vector"].value_counts().to_dict(),
        "path_x_arch": {
            "vectorizador": str(vect_path),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv)
        }
    }

    meta_path = output_dir / f"preprocesamiento_metadata_{version}.json"
    
    print(f"\nGuardando metadatos en: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:     #Abrimos el fichero JSON en modo escritura con codificacion utf-8 mediante el with
        json.dump(m, f, indent=4, ensure_ascii=False)       #Generamos el JSON con el formato con indent=4 y ensure_ascii=False

    print("\nArtefactos guardados")
    print("Vectorizador:", vect_path)
    print("Train:", train_csv)
    print("Test:", test_csv)
    print("Meta:", meta_path)
    print("\n")