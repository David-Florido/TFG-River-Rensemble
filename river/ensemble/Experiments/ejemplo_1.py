from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

dataset = datasets.Phishing()

base_classifier =  preprocessing.StandardScaler() | linear_model.LogisticRegression()
meta_classifier = preprocessing.StandardScaler() | linear_model.LogisticRegression()

models_3x5   =  [(3, (base_classifier, 5))]

model = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=42, unanimity_check=False, drift_check = "off")

metric = metrics.F1()
print(evaluate.progressive_val_score(dataset, model, metric))