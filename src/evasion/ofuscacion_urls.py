import random
import string
from urllib.parse import urlparse, urlunparse, quote

def percent_encoding_url(url: str, prob: float = 0.3) -> str:

    #Se aplica la tecnica percent-encodig de manera parcial sobre algunos caracteres de la URL

    try:
        url_descompuesta = urlparse(url)             #Se divide la URL en sus partes principales
    except Exception:
        return url
    
    #Se ofusca la parte path
    new_path = "".join(
        quote(carac) if (carac.isalnum() and random.random() < prob) else carac     #Se verifica si es un alfanumerico y si el nuemro aleatorio es menor que prob, si se cumple ofusca aleatoriamente al convertir el carcter en URL-encoded
        for carac in url_descompuesta.path                                          #Se recorre la ruta caracter por caracter
    )

    #Se ofsuca la parte query
    new_query = "".join(
        quote(carac) if (carac.isalnum() and random.random() < prob) else carac
        for carac in url_descompuesta.query
    )

    #Se reemplaza las nuevas partes por las antiguas y se unifica
    new_url = url_descompuesta._replace(path=new_path, query=new_query)
    return urlunparse(new_url)

def variaciones_dominio(dominio: str) -> str:

    #Se aplican variaciones simples al dominio, tales como duplicidades, eliminacion e intercambio de letras

    if len(dominio) < 3:
        return dominio
    
    acc = random.choice(["duplicar", "eliminar", "swap"])       #Se elige una operacion aleatoria
    i = random.randint(0, len(dominio) - 2)                     #Se elige un indice aleatorio valido

    caracteres = list(dominio)

    if acc == "duplicar":
        caracteres.insert(i, caracteres[i])
    elif acc == "eliminar":
        del caracteres[i]
    elif acc == "swap":
        caracteres[i], caracteres[i+1] = caracteres[i+1], caracteres[i]
    
    return "".join(caracteres)

def ofuscar_dominio(url: str) -> str:

    #Se aplican las variaciones descritas en el modulo anterior

    try:
        url_descompuesta = urlparse(url)
    except:
        return url
    
    d = url_descompuesta.netloc       #Obtenemso el dominio de la url
    if not d:
        return url
    
    new_dominio = variaciones_dominio(d)
    new_url = url_descompuesta._replace(netloc=new_dominio)
    return urlunparse(new_url)

def añadir_subdominios(url: str) -> str:

    #Se añade subdominios sospechosos que parecen legitimos

    try:
        url_descompuesta = urlparse(url)
    except Exception:
        return url
    
    d = url_descompuesta.netloc
    if not d or "." not in d:
        return url
    
    prefijos = ["login", "support", "update", "seguro", "academy"]
    new_prefix = random.choice(prefijos)

    new_dominio = f"{new_prefix}.{d}"
    new_url = url_descompuesta._replace(netloc=new_dominio)
    return urlunparse(new_url)

def aplicar_grado_ofuscacion(url: str, nivel: str = "medio") -> str:

    #Se definen unos nivels de ofuscacion para posteriormente aplicar una combinacion de ofuscacion regulada segun el nivel.

    t = url

    if nivel == "suave":
        if random.random() < 0.5:
            t = percent_encoding_url(t, prob=0.05)
        
        return t
    
    elif nivel == "fuerte":

        if random.random() < 0.9:
            t = percent_encoding_url(t, prob=0.40)
        
        if random.random() < 0.85:
            t = ofuscar_dominio(t)

        if random.random() < 0.6:
            t = añadir_subdominios(t)

        return t
        
    else:   #Equivale al medio siendo el predefinido

        if random.random() < 0.7:
            t = percent_encoding_url(t, prob=0.20)
        
        if random.random() < 0.85:
            t = ofuscar_dominio(t)

        return t
