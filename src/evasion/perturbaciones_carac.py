import random
import re

#Se instancia los homoglyphs siendo estos visualmente similares
HOMOGLYPHS = {
    "a": ["а", "à", "á"],        # la primera es a cirílica
    "c": ["с"],                  # la c es cirílica
    "e": ["е", "è", "é"],        # la primera es cirílica
    "i": ["і", "í"],             # la primera es i cirílica
    "o": ["о", "ò", "ó"],        # la primera es o cirílica
    "p": ["р"],                  # la p es cirílica
    "x": ["х"],                  # la x es cirílica
    "y": ["у"],                  # la y es cirílica
    "u": ["υ", "ù", "ú"],        # también incluye upsilon griega
    "l": ["ⅼ", "1"],             # letra romana + número 1
    "0": ["Ο", "O"],             # omicron griega + O mayúscula
}

def insertar_ZeroWidth_space(texto: str, prob: float = 0.1) -> str:

    #Se inserta espacios invisibles (zero-width space) en el texto con probabilidad por carcter

    ZWSP = "\u200b"     #zero-width space
    result = []

    #Se recorre el texto de caracter a caracter para ir contruyendo una nueva cadena
    for carac in texto:
          result.append(carac)
          if carac.isalnum() and random.random() < prob:      #Si el caracter es alfanumerico, se inserta un zero-width con cierta probabilidad
               result.append(ZWSP)

    return "".join(result)

def sustituir_homoglyphs(texto: str, prob: float = 0.2) -> str:
     
    #Se sustituye caracteres por homologlyphs con cierta probabilidad y de manera aleatoria
    
    result = []

    #Se recorre el texto de caracter a caracter para ir convirtiendo en minuscula e ir construyendo una nueva cadena
    for carac in texto:
        minus = carac.lower()
        if minus in HOMOGLYPHS and random.random() < prob:  #En caso de aparecer en el diccionario y si el numero aleatorio es menor que prob.
            sustituto = random.choice(HOMOGLYPHS[minus])    #Se escoge una de las opciones en el diccionario

            if carac.isupper():                             #Se conserva la mayuscula en caso de haber sido el original
                 sustituto = sustituto.upper()

            result.append(sustituto)
        else:
            result.append(carac)
    
    return "".join(result)

def introducir_errores_tipograficos(texto: str, prob: float = 0.05) -> str:
    #Se introducen errores tipograficos simples como duplicar letras o intercambiar orden

    caracteres = list(texto)
    i = 0

    while i < len(caracteres) - 1:
        if caracteres[i].isalpha() and random.random() < prob:      #Se verifica si es una letra del alfabeto y si el nuemro aleatorio es menor que prob
            acc = random.choice(["duplicar", "swap"])
            if acc == "duplicar":
                caracteres.insert(i, caracteres[i])
                i += 2
                continue
            elif acc == "swap":
                caracteres[i], caracteres[i+1] = caracteres[i+1], caracteres[i]
                i += 2
                continue
        i += 1

    return "".join(caracteres)

def aplicar_grado_perturbaciones(texto: str, nivel: str = "medio") -> str:

    #Se definen unos nivels de perturbacion para posteriormente aplicar una combinacion de perturbaciones regulada segun el nivel.

    if nivel == "suave":
        prob_zw = 0.05
        prob_homoglyph = 0.1
        prob_err_tipograficos = 0.02
    elif nivel == "fuerte":
        prob_zw = 0.2
        prob_homoglyph = 0.4
        prob_err_tipograficos = 0.1
    else:   #Equivale al medio siendo el predefinido
        prob_zw = 0.1
        prob_homoglyph = 0.2
        prob_err_tipograficos = 0.05

    t = texto
    t = insertar_ZeroWidth_space(t, prob=prob_zw)
    t = sustituir_homoglyphs(t, prob=prob_homoglyph)
    t = introducir_errores_tipograficos(t, prob=prob_err_tipograficos)

    return t


        
