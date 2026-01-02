import shutil
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

#Instancia de constantes
CANT_VECT_MAL_LEG = 3000       #Cantidades de muestras por vector (6k en total por cada uno (3k de legitimas y 3k de maliciosas))
RANDOM_STATE = 42              #Semilla para garantizar la reproducibilidad del experimento, asegurando la trazabilidad del pipeline de datos
DATASET_VERSION = "v2"
DATASET_VERSION_ANTERIOR = "v1"

#RUTAS
BASE_DIR = Path(__file__).resolve().parents[2] #Se recoje la ruta actual y sube dos niveles
ORIG_DIR = BASE_DIR / "data" / "orig"
UNIFIC_DIR = BASE_DIR / "data" / "unificado"
UNIFIC_DIR.mkdir(parents=True, exist_ok=True) #Se crea la carpeta si no existe y no da error si ya existia

#RUTAS .csv
RUTA_URLS = ORIG_DIR / "phishing_url.csv"
RUTA_EMAILS = ORIG_DIR / "phishing_email.csv"
RUTA_SMS = ORIG_DIR / "phishing_smishing.csv"
OUTPUT = UNIFIC_DIR / f"dataset_unificado_{DATASET_VERSION}.csv"
JSON_OUTPUT = UNIFIC_DIR / f"dataset_unificado_{DATASET_VERSION}_meta.json"
LATEST_OUTPUT = UNIFIC_DIR / "dataset_unificado_latest.csv"

#Mediante esta funcion se pretende dovolver un subconjunto balanceado igual a CANT_VECT_MAL_LEG en caso de tener lso suficientes
#en caso contrario cogera todo las muestras que contenga este.

def bal_x_clase(d, lab_col="label",n_x_clase=3000, rs = 42):
    cl = d[lab_col].dropna().unique()   #Obtiene las clases unicas del data sin incluir los valores vacios
    mue = []

    for i in cl:
        d_iterativa = d[d[lab_col] == i]
        min_mues = min(n_x_clase, len(d_iterativa))     #Se queda con el minimo en caso de que no haya muestras suficientes

        if min_mues == 0:       #Salta en caso de no tener muestras en lugar de dar error.
            continue

        mue.append(d_iterativa.sample(n=min_mues, random_state=rs)) 

    if not mue:
        raise ValueError("No se ha podido extraer las muestras")    #Se lanza error en caso de no extraer nada
    
    return pd.concat(mue, ignore_index=True)       #Se devuleve el dataset balanceado

def cargar_datos_urls():
    d = pd.read_csv(RUTA_URLS)

    d.columns = d.columns.str.strip()       #Se asegura de que las columnas no tenga espacios

    d = d.rename(columns={"URL": "texto"})
    d["vector"] = "url"
    d["fuente"] = "phiusiil"

    d["label"] = d["label"].astype(int)     #Se asegura que sea de tipo entero
    d["label"] = 1 - d["label"]             #Se intercambia los valores para phishing y legitimo para coindir con los otros datasets

    #Se devuelven las columnas que necesitamos y, sobretodo que no tengan valores vacios en texto y label
    return d[["texto", "label", "vector", "fuente"]].dropna(subset=["texto", "label"])

def cargar_datos_email():
    d = pd.read_csv(RUTA_EMAILS)

    d.columns = d.columns.str.strip()       #Se asegura de que las columnas no tenga espacios

    d = d.rename(columns={"text_combined": "texto"})
    d["vector"] = "email"
    d["fuente"] = "kaggle_email"

    d = d.dropna(subset=["texto", "label"])

    d["label"] = d["label"].astype(int)

    return d[["texto", "label", "vector", "fuente"]]

def cargar_datos_sms():
    d = pd.read_csv(RUTA_SMS)

    d.columns = d.columns.str.strip()       #Se asegura de que las columnas no tenga espacios

    d = d.rename(columns={"TEXT": "texto", "LABEL": "label"})
    d["vector"] = "sms"
    d["fuente"] = "mendeley"

    d["label"] = d["label"].astype(str).str.lower().str.strip()

    mapa = {
        "ham": 0,
        "spam": 1,
        "smishing": 1
    }

    d["label"] =d["label"].map(mapa)

    d = d.dropna(subset=["texto", "label"])

    d["label"] = d["label"].astype(int)

    return d[["texto", "label", "vector", "fuente"]]

def main():
    print("Cargando datasets...")
    dt_urls = cargar_datos_urls()
    dt_emails = cargar_datos_email()
    dt_sms = cargar_datos_sms()

    #Mostramos la cantidad de ejemplos que hay por cada uno y lo muestra como diccionario
    print(f"URLs: {dt_urls['label'].value_counts().to_dict()}")
    print(f"Emails: {dt_emails['label'].value_counts().to_dict()}")
    print(f"SMS: {dt_sms['label'].value_counts().to_dict()}")


    print(f"\nBalanceados por clase ({CANT_VECT_MAL_LEG} legitimos + {CANT_VECT_MAL_LEG} maliciosos por vector si existen suficientes)")
    dt_urls_bal = bal_x_clase(dt_urls, lab_col="label", n_x_clase=CANT_VECT_MAL_LEG, rs=RANDOM_STATE)
    dt_email_bal = bal_x_clase(dt_emails, lab_col="label", n_x_clase=CANT_VECT_MAL_LEG, rs=RANDOM_STATE)
    dt_sms_bal = bal_x_clase(dt_sms, lab_col="label", n_x_clase=CANT_VECT_MAL_LEG, rs=RANDOM_STATE)

    print(f"URLs balanceado: {dt_urls_bal.shape}")
    print(f"Emails balanceado: {dt_email_bal.shape}")
    print(f"SMS balanceado: {dt_sms_bal.shape}")

    print("\nUnificando datasets....")
    dt_unificado = pd.concat([dt_urls_bal, dt_email_bal, dt_sms_bal], ignore_index=True)

    #Se mezcla los datos de manera aleatoria con la semilla
    dt_unificado = dt_unificado.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    dt_unificado["id"] = dt_unificado.index

    print(f"Dataset final: {dt_unificado.shape}")
    print(dt_unificado.head())

    #se convierte las claves del groupby en string para el formato de JSON
    mue_vec_label = (
        dt_unificado.groupby(["vector", "label"]).size().reset_index(name="count")
    )

    #Se crea el diccionario donde habra las claves string y se reemplaza index por _ porque index existe pero no es redundante
    mue_vec_label_dict = {
        f"{row['vector']}__{row['label']}": int(row["count"])
        for _, row in mue_vec_label.iterrows()
    }

    #Se prepara los metadatos para la trazabilidad con un JSON
    m = {
        "version": DATASET_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "total_samples": int(len(dt_unificado)),
        "samples_by_label": dt_unificado["label"].value_counts().to_dict(),
        "samples_by_vector": dt_unificado["vector"].value_counts().to_dict(),
        "samples_by_vector_and_label": mue_vec_label_dict,
        "classes_definition": {
            "label": {
                "0": "legitimo",
                "1": "phishing/smishing"
            }
        },
        "sources": sorted(dt_unificado["fuente"].unique().tolist()),
        "config": {
            "cantidad_x_vector": CANT_VECT_MAL_LEG,
            "orig_dir": str(ORIG_DIR),
            "unific_dir": str(UNIFIC_DIR),
        }
    }

    print(f"\nGuardando versio anterior en: {OUTPUT} (si existe una anterior)")

    if LATEST_OUTPUT.exists():
        backup = OUTPUT.with_name(f"dataset_unificado_{DATASET_VERSION_ANTERIOR}.csv")
        shutil.copy2(LATEST_OUTPUT, backup)
        print(f"\nVersion anterior copiada a {backup}")
    else:
        print("No existe la anterior version que copiar")

    dt_unificado.to_csv(OUTPUT, index=False)
    print(f"\nVersion nueva guardada en: {OUTPUT}")

    dt_unificado.to_csv(LATEST_OUTPUT, index=False)
    print(f"\nVersion ultima actualizada en: {LATEST_OUTPUT}")

    print(f"\nGuardando metadatos en: {JSON_OUTPUT}")
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:     #Abrimos el fichero JSON en modo escritura con codificacion utf-8 mediante el with
        json.dump(m, f, indent=4, ensure_ascii=False)       #Generamos el JSON con el formato con indent=4 y ensure_ascii=False

    print("\nGuardado")


if __name__ == "__main__":
    main()