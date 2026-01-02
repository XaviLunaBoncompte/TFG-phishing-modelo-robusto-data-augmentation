from __future__ import annotations
import random
import re
import numpy as np
from typing import Any, Mapping

from ..evasion.perturbaciones_carac import aplicar_grado_perturbaciones
from ..evasion.ofuscacion_urls import aplicar_grado_ofuscacion
from ..evasion.parafraseo_semantico import aplicar_variantes_semanticas
from ..evasion.template_mixing import generar_template_mixing

from ..config.rutas import RANDOM_STATE

def fijar_semillas(seed: int = RANDOM_STATE) -> None:

    #Se fija la semilla en 42 para la reproducibilidad en tecnicas que useen random

    random.seed(seed)
    np.random.seed(seed)

def generar_variantes_evasivas(
    row: Any,
    nivel: str = "medio",
    prob_semantica: float = 0.3,
    prob_template: float = 0.6,
    prob_perturbacion: float = 0.8
) -> str:
    
    #Se aplica variantes evasivas sobre filas que contengan texto_preprocesado y/o vector

    texto = row["texto_preprocesado"]
    v = row["vector"].lower()

    #Para muestras de tipo URL
    if v == "url":
        return aplicar_grado_ofuscacion(texto, nivel=nivel) if random.random() < 0.9 else texto
    
    t = texto

    #Para el resto de muestras que no sean URL
    if random.random() < prob_perturbacion:
        t = aplicar_grado_perturbaciones(t, nivel=nivel)
    
    if random.random() < prob_template:
        t = generar_template_mixing(t, nivel=nivel)

    if random.random() < prob_semantica:
        t = aplicar_variantes_semanticas(t, metodo="back_translation")

    return t