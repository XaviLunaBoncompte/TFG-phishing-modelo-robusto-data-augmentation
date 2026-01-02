.: DESCRIPCIÓN :.
El presente prototipo cuya finalidad es de investigación, tiene el objetivo de construir, evaluar y reforzar un sistema de detección de phishing empleando aprendizaje automatico junto con generacion de variantes evasivas, las cuales simulen el comportamiento de intelifencia artificial ofensiva.

El enfoque empleado se basa en transformar un conjunto de muestras en vectores numericos a traves de TF-IDF, para posteriormente entrenar los modelos capaces de detectar contenido malicioso.

Para ello, el pipeline actual incluye:
- Unificacion de los datasets con mas de 16.000 muestras.
- Analaisis exploratorio (EDA).
- Limpieza ligera y preprocesamiento.
- Division en subconjuntos de entrenamiento y test.
- Vectorizacion mediante TF-IDF.
- Modelos base (SVM) y complememntario (MLP + reduccion dimensional con SVD).
- Ensemble mediante Hard Voting.
- Evaluacion inicial sobre el dataset unificado.
- Generacion de variantes evasivas.
- Trazabilidad y explicabilidad
- Evaluacion final.

.: OBJETIVOS :.
Diseñar y validará un prototipo práctico, basado en el aprendizaje automático capaz de mejorar la detección de amenazas de tipo phishing cada vez más sofisticadas. 

.: DIRECTORIOS :.

/config --> configuraciones de rutas para utilizar mas tarde
/data --> datasets originales, procesados y unificados guardados con sus versiones
/documentacion --> Memoria y documentacion complementaria
/models --> metadatos y subconjuntos guardados mediante joblib, csv o json para trazabilida y explicabilidad
/notebooks --> Interacción con el codigo de manera interativa por cada modulo/fase
/src --> Codigo del proyecto con el main y modulos

.: DATASET :.

El dataset esta compuesto por un conjunto unificado de muestras procedentes de tres fuentes fiables, las cuales contienen muestras maliciosas y legitimas. Tras la unificacion y limpieza, el dataset contiene un total de 16.126 muestras y 6 columnas principales, tales como: text, label (0 = legitimos o 1 = maliciosos), vector (email, url, sms), fuente (procedencia), id y texto_preprocesado.

.: EJECUCIÓN DEL PROYECTO :.
1. Descomprimir o clonar el proyecto, y abrirlo con VS Code
2. Ir a la Terminal (Ver > Terminal) y crear un entorno virtual venv e instalar dependencias:

    Desde la terminal hay que ejecutar los comandos:

    - Crear entorno virtual --> pyhton -m venv venv

    - Activar el entorno virtual --> Ubicarse en el directorio raiz TFG "..\TFG_prototipo_xlunab"--> Windows: venv\Scripts\activate
                                                                                     Linux/Mac: source venv/bin/activate

    - Instalar dependencias --> pip install -r requeriments.txt y pip install -r requeriments_dev.txt

3. Ejecucion del proyecto:

El proyecto completo y evaluacion se ubican en la carpeta /src. 

Antes de nada hay que cambiar la version que tendra nombrada el dataset en config/rutas.py si es el caso de que hay cambios, y posteriormente ejecutar el siguiente comando en el entorno virtual venv para unificar el dataset. 

    python -m src.data_prepare.cons_dataset_unificado 

Una vez obtenido el dataset unificado hay entrenar el modelo base y el robusto con evasion

- Base --> pyhton -m src.models.modelado 
- Robusto (con evasion) --> python -m src.models.modelado_evasivo 

Se generaran los modelos entrenados como modelo_nombre_version.joblib y los correspondientes tfidf, train, test, etc.

4. Evaluacion de robustez (base vs robusto)

Hay que ejecutar el siguente comando en el que se evalua 4 escenarios y se generan metricas comapartivas en formato JSON (metricas_robustez_*(ID_Run)_versionbase_vs_versionrobusta)

    pyhton -m src.models.evaluacion_robustez

5. Los metadatos y versiones se guardan en C:/TFG/models


.: ANALISIS Y EXPERIMENTOS INDIVIDUALES :.

El presente trabajo tambien cuenta con una serie de notebooks con Jupyter, los cuales han servido como introductorios para testear nuevas implementaciones o analizar datos respecto lo implementado. Su exploracion esta disponible una vez creado el entorno virtual, siendo ejecutables cuando se selecciona este como Kernel.

Una vez seleccioando el Kernel, simplemente hay que ejecutar todo o celda por celda. En caso de no cargar hay que interrumpir y reinciar el kernel.








Este proyecto se entrega explusivamente con fines academicos como parte del TFG.