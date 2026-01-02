from __future__ import annotations

from functools import lru_cache
from typing import List

from transformers import pipeline

@lru_cache(maxsize=1)           #Con lru_cache maxsize = 1 solo se carga una vez el modulo
def obt_traductor_en_es():

    #Se instancia el modulo traductor de Ingles a Español

    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

@lru_cache(maxsize=1)           #Con lru_cache maxsize = 1 solo se carga una vez el modulo
def obt_traductor_es_en():

    #Se instancia el modulo traductor de Español a Ingles

    return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def back_translation_en(texto: str, max_carac: int = 1200) -> str:

    #Aplica back-translation sobre un texto en ingles, manteniendo la sematica pero alterando el lexico

    if not isinstance(texto, str) or not texto.strip():
        return texto

    t = texto.strip()
    if len(t) > max_carac:
        t = t[:max_carac]

    try:
        #De Ingles a Español seleccionando la primera traduccion de la lista
        trad_en_es = obt_traductor_en_es()
        texto_traducido_a_esp = trad_en_es(t, max_length=256, truncation=True)[0]["translation_text"]

        #De Español a Ingles de nuevo, tambien seleccionando la primera traduccion de la lista
        trad_es_en = obt_traductor_es_en()
        resultado_back = trad_es_en(texto_traducido_a_esp, max_length=256, truncation=True)[0]["translation_text"]

        return resultado_back
    
    except Exception:
        return t




@lru_cache(maxsize=1)
def obt_parafraseo_en():

    #Se devuelve un pipeline de parafraseo basado en T5 para textos en ingles

    return pipeline(
        "text2text-generation",
        model="t5-base"
    )

def parafraseo_en(texto: str, num_return_sequences: int = 1, num_beams: int = 4) -> List[str]:

    #Se genera una o varias parafrasis de un texto en ingles usando T5

    if not isinstance(texto, str) or not texto.strip():
        return [texto]
    
    parafraseo = obt_parafraseo_en()        #Se carga el modelo t5
    prefijo = f"paraphrase the following sentence: {texto}"        #Se le instancia el formato/prompt requerido para que el modelo reescriba manteniendo el significado

    salida = parafraseo(                    #SE genera la salida con una o varias parafrasis
        prefijo,
        max_length=128,
        num_return_sequences=num_return_sequences,      #Numero de sentencias a devolver
        num_beams=num_beams,                            #Se explora num_beans reformulaciones
        clean_up_tokenization_spaces=True               #Limpia espacios raros
    )

    variantes = [i["generated_text"] for i in salida]
    return variantes

def aplicar_variantes_semanticas(texto: str, metodo: str = "back_translation") -> str:

    #Se aplica la metodoliga a emplear segun lo que se introduc en la variable metodo.
    
    if metodo == "t5":
        return parafraseo_en(texto, num_return_sequences=1)[0]      #Se genera una sola variante y devuelve el primer resultado
    else:
        return back_translation_en(texto)

