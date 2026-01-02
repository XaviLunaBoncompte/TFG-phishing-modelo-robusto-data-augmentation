import re
import pandas as pd
from pathlib import Path

def limpiar_texto(t: str) -> str:

    #Se realiza una limpieza ligera y segura para los tres tipos de vecrores principales preservando la semantica
    if not isinstance(t, str):
        t = str(t)
    
    t = t.lower()                           #Convertir a minusculas
    t = re.sub(r"\s", " ", t)               #Elimina saltos de linea
    t = re.sub(r"[^\x20-\x7E]", "", t)      #Elimina carcteres no imprimibles
    t = t.strip()                           #Compacta los espacios

    return t

def preprocesar_dataset(dt: pd.DataFrame) -> pd.DataFrame:

    #Aplica la funcion de limpieza y elimina los textos vacios despues de limpiar.

    dt = dt.copy()

    dt["texto_preprocesado"] = dt["texto"].apply(limpiar_texto)

    dt = dt[dt["texto_preprocesado"].str.len() > 0].reset_index(drop=True)

    return dt

