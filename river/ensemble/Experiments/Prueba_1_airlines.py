import os
import time
import copy
from river import naive_bayes
from river import tree
from river import ensemble

from experiment_functions import test_then_train, loadAirlines

os.makedirs(os.path.dirname("./results/plots/time/"), exist_ok=True)
os.makedirs(os.path.dirname("./results/plots/memory/"), exist_ok=True)
os.makedirs(os.path.dirname("./results/plots/accuracy/"), exist_ok=True)

accuracy = {}
loaders = [loadAirlines]
dataset_names = ["airlines_prueba_1"]

for index in range(len(loaders)):
    start_time = time.time()
    dataset = loaders[index]()
    dataset_name = dataset_names[index]

    base_classifier = naive_bayes.MultinomialNB()

    meta_classifier = tree.HoeffdingTreeClassifier(
        grace_period=50, split_confidence=0.01,
    )
    
    models_STACK =  [(1, (base_classifier,19))]
    models_3x5   =  [(3, (base_classifier, 5))]
    models_6x2   =  [(6, (base_classifier, 2))]
    models_cfc   =  [(10,(base_classifier, 1))]

    BAG = ensemble.BaggingClassifier(
        base_classifier,
        n_models=20,
        seed=13
    )

    STACK =  ensemble.StackingClassifier(
        [copy.deepcopy(base_classifier) for _ in range(20)],
        meta_classifier
    )

    SRP = ensemble.SRPClassifier(
        model=base_classifier, n_models=20, seed=13
    )

    R_STACK = ensemble.RensemblerClassifier(models_STACK, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")
    R_3x5 = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")
    R_6x2 = ensemble.RensemblerClassifier(models_6x2, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")
    CFC = ensemble.RensemblerClassifier(models_cfc, meta_classifier, lam=1.0, seed=13, unanimity_check=False, drift_check = "off")

    model_list = [BAG, STACK, SRP, R_STACK, R_3x5, R_6x2, CFC]
    model_name_list = ["BAG", "STACK", "SRP", "R_STACK", "R_3x5", "R_6x2", "CFC"]
    test_then_train(dataset, model_list, model_name_list, dataset_name)

    end_time = time.time()
    execution_time = end_time - start_time

    text_file = open(f"./results/{dataset_name}_execution_time.txt", "w")
    text_file.write(f"Execution time:\t{execution_time} seconds")
    text_file.close()
    print(f"\nSUCCESS!\nExecution time:\t{execution_time} seconds")