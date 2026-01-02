from __future__ import annotations

import random
from typing import Dict

INTRO_TEXTO_LEGITIMO = [
    "Hello,",
    "Dear customer,",
    "Estimado cliente,",
    "Nos ponemos en contacto con usted para informarle de una novedad importante."
]

CUERPO_TEXTO_LEGITIMO = [
    "Se trata de un aviso generado automaticamente por nuestro sistema de seguridad,",
    "This is an automatic notification from our security system.",
]

CIERRE_TEXTO_LEGITIMO = [
    "Gracias por su confianza.",
    "Best regards, \nSecurity team",
    "Atentamente, \nDepartamento de seguridad",
    "Thank you for your cooperation.",
    "Si ya ha realizado esta accion, puede ignorar este correo"
]

def generar_template_mixing(texto_malicioso: str, nivel: str = "medio", contexto: Dict | None = None) -> str:

    #Se generan mensajes mixtos combinando los fragmentos benignos instanciados en los diccionarios junto con texto maligno original.

    if not isinstance(texto_malicioso, str) or not texto_malicioso.strip():
        return texto_malicioso
    
    intro = random.choice(INTRO_TEXTO_LEGITIMO)
    cuerpo = random.choice(CUERPO_TEXTO_LEGITIMO)
    cierre = random.choice(CIERRE_TEXTO_LEGITIMO)

    if nivel == "suave":
        mensaje = f"{intro} {texto_malicioso}"
    elif nivel == "fuerte":
        intro_extra = random.choice(INTRO_TEXTO_LEGITIMO)
        cierre_extra = random.choice(CIERRE_TEXTO_LEGITIMO)

        mensaje = (
            f"{intro}\n\n"
            f"{cuerpo}\n\n"
            f"{texto_malicioso}\n\n"
            f"{intro_extra}\n\n"
            f"{cierre}\n\n"
            f"{cierre_extra}"
        )
    else:
        mensaje = (
            f"{intro}\n\n"
            f"{cuerpo}\n\n"
            f"{texto_malicioso}\n\n"
            f"{cierre}"
        )
        
    return mensaje
