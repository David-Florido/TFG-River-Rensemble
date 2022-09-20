from pprint import pprint
from river import datasets

dataset = datasets.Phishing()

for x, y in dataset:
    pprint(x)
    print(y)
    break

from river import naive_bayes
from river import tree
from river import ensemble
from river import metrics

base_classifier = naive_bayes.MultinomialNB()

meta_classifier = tree.HoeffdingTreeClassifier(
    grace_period=50, split_confidence=0.01,
)
    
base_models   =  [(3, (base_classifier, 5))]

model = ensemble.RensemblerClassifier(base_models, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")

metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model.learn_one(x, y)      # make the model learn

print(metric)