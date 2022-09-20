import os
import time
import copy
from river import naive_bayes
from river import tree
from river import ensemble

from experiment_functions import test_then_train, loadRBF

os.makedirs(os.path.dirname("./results/plots/time/"), exist_ok=True)
os.makedirs(os.path.dirname("./results/plots/memory/"), exist_ok=True)
os.makedirs(os.path.dirname("./results/plots/accuracy/"), exist_ok=True)

accuracy = {}
n = 500000
loaders = [loadRBF]
dataset_names = ["RBF_prueba_2"]

for index in range(len(loaders)):
    start_time = time.time()
    dataset = loaders[index](n)
    dataset_name = dataset_names[index]

    base_classifier = naive_bayes.MultinomialNB()

    meta_classifier = tree.HoeffdingTreeClassifier(
        grace_period=50, split_confidence=0.01,
    )

    models_3x5   =  [(3, (base_classifier, 5))]

    R_Basic = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")
    R_Unanimity = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=13, unanimity_check=0.8, drift_check = "off")
    R_Model = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "model")
    R_Module = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "module")

    model_list = [R_Basic, R_Unanimity, R_Model, R_Module]
    model_name_list = ["Basic", "Unanimity", "Model", "Module"]
    test_then_train(dataset, model_list, model_name_list, dataset_name)

    end_time = time.time()
    execution_time = end_time - start_time

    text_file = open(f"./results/{dataset_name}_execution_time.txt", "w")
    text_file.write(f"Execution time:\t{execution_time} seconds")
    text_file.close()
    print(f"\nSUCCESS!\nExecution time:\t{execution_time} seconds")