## ❓ Resumen del sistema
El sistema consiste en una generalización de sistemas multiclasificadores. En él, se generan m módulos, cada uno con n clasificadores base, 1 meta-clasificador. Cada instancia será procesada de una en una por los clasificadores base, cuya predicción será añadida al resto de características y transmitida como entrada al meta-clasificador del módulo. Posteriormente se decidirá la respuesta final del sistema mediante voto ponderado de las predicciones emitidas por cada uno de estos meta-clasificadores.

![esquema](https://github.com/David-Florido/TFG-River-Rensemble/blob/main/Esquema%20del%20Sistema.png?raw=true)

## 🛠 Instalación
Las pruebas se han realizado con **Python 3.10.4**, el cual se puede obtener en [este enlace](https://www.python.org/downloads/)
Adicionalmente, para utilizar la librería **River** es necesario tener instalados [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


Para instalar el código una vez descargado, basta con ejecutar:
```sh
pip install -e {PATH} --user
```
donde PATH es la ubicación hasta el directorio descomprimido "TFG-River-Rensemble-main".

## ⚡️  Guía rápida
Como ejemplo, utilizaremos el [dataset de phishing](http://archive.ics.uci.edu/ml/datasets/Website+Phishing). Aquí añadimos el código necesario para cargarlo y observar la primera instancia:

```python
from pprint import pprint
from river import datasets

dataset = datasets.Phishing()

for x, y in dataset:
    pprint(x)
    print(y)
    break
    
{'age_of_domain': 1,
 'anchor_from_other_domain': 0.0, 
 'empty_server_form_handler': 0.0,
 'https': 0.0,
 'ip_in_url': 1,
 'is_popular': 0.5,
 'long_url': 1.0,
 'popup_window': 0.0,
 'request_from_other_domain': 0.0}
True
```

Una vez tenemos el dataset en memoria, crearemos los modelos que usaremos para la clasificación, para ello, especificamos el tipo de clasificador base, configuración, y meta-clasificador a utilizar.
```python
from river import naive_bayes
from river import tree
from river import ensemble

base_classifier = naive_bayes.MultinomialNB()

meta_classifier = tree.HoeffdingTreeClassifier(
    grace_period=50, split_confidence=0.01,
)

base_models   =  [(3, (base_classifier, 5))]

model = ensemble.RensemblerClassifier(base_models, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")
```

Tras haber generado el modelo a utilizar, podemos comprobar su funcionamiento:
```python
from river import metrics
metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)      # realiza predicción
    metric = metric.update(y, y_pred)  # actualiza precisión
    model = model.learn_one(x, y)      # entrena el modelo

print(metric)
Accuracy: 87.76%
```

Esto es un ejemplo sencillo de uso, para información más detallada sobre su funcionamiento, significado de parámetros se puede consultar la [clase principal](river/ensemble/rensemble.py). Pruebas y ejemplos se pueden localizar en [este directorio](river/ensemble/Experiments). 
Para ejecutar el código usado en la experimentación es necesario instalar las siguientes librerías:
```sh
pip install sklearn
pip install matplotlib
```
