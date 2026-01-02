from .perturbaciones_carac import (
    aplicar_grado_perturbaciones
)

from .ofuscacion_urls import (
    aplicar_grado_ofuscacion
)

from .template_mixing import (
    generar_template_mixing
)

from .parafraseo_semantico import (
    aplicar_variantes_semanticas
)

__all__ = [
    aplicar_grado_perturbaciones,
    aplicar_grado_ofuscacion,
    generar_template_mixing,
    aplicar_variantes_semanticas
]