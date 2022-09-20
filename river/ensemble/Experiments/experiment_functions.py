import time
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from math import ceil
from sklearn import preprocessing
from river import synth
import re


def test_then_train(data, model_list, model_name_list, dataset_name):
    timestamps = {}
    memory_usages = {}
    accuracy = {}

    acc_text = open(f"./results/{dataset_name}_accuracy.txt", "w")
    size = len(data)
    step_size = ceil(size/10)
    step_size
    lower_idx = 0
    upper_idx = step_size + size % 10
    print(f"Dataset: {dataset_name}")
    for step in range(10):
        print(f"\nStep {step+1} ({(step+1)*10}%)\n")
        acc_text.write(f"\nStep {step+1} ({(step+1)*10}%)\n")
        training_data = data[lower_idx : upper_idx]

        for i in range(len(model_list)):
            model = model_list[i]
            model_name = model_name_list[i]
            print(f"Model {i}: {model_name}")

            total = 0
            correct = 0
            start_time = time.time()
            for x,y in training_data:

                y_pred = model.predict_one(x)
                total += 1
                if y_pred == y:
                    correct += 1

                model.learn_one(x,y)
                    
            end_time = time.time()

            timestamps[i] = timestamps.get(i, 0) + (end_time - start_time)
            memory_usage = float(re.search(r'\d+(\.\d+)?', model._memory_usage).group(0))
            unit = re.search(r'[KMG]B', model._memory_usage).group(0)
            if unit == "KB":
                memory_usage *= 0.001
            elif unit == "GB":
                memory_usage *= 1000
            memory_usages[i] = memory_usages.get(i, 0) if memory_usages.get(i, 0) > memory_usage else memory_usage
            if not dataset_name in accuracy:
                accuracy[dataset_name] = {}
            if not model_name in accuracy[dataset_name]:
                accuracy[dataset_name][model_name] = []

            accuracy[dataset_name][model_name].append(correct/total)
            acc_text.write(f"(MODEL {model_name}) {correct} correct out of {total} ({correct/total} accuracy)\n")

        lower_idx = upper_idx
        upper_idx = min(upper_idx + step_size, size)

    acc_text.close()

    time_text = open(f"./results/{dataset_name}_timestamps.txt", "w")
    for i in timestamps:
        time_text.write(f"{model_name_list[i]}: {timestamps[i]}\n\n")
        plt.ylabel('time (s)')
        plt.bar(model_name_list[i], timestamps[i])
        plt.grid(visible=True)
    plt.savefig(f'./results/plots/time/{dataset_name}.png')
    time_text.close()
    plt.clf()
    
    mem_text = open(f"./results/{dataset_name}_memory_usage.txt", "w")
    for i in memory_usages:
        mem_text.write(f"{model_name_list[i]}: {memory_usages[i]}\n\n")
        plt.ylabel('max memory usage (MB)')
        plt.bar(model_name_list[i], memory_usages[i])
        plt.grid(visible=True)
    plt.savefig(f'./results/plots/memory/{dataset_name}.png')
    mem_text.close()
    plt.clf()

    for model in accuracy[dataset_name]:
        x = [10,20,30,40,50,60,70,80,90,100]
        plt.plot(x, accuracy[dataset_name][model], label=model)
        plt.xticks(x)
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('% of dataset used')
        plt.ylabel('accuracy')
        plt.grid(visible=True)
        plt.ylim([0, 1.1])

    plt.savefig(f'./results/plots/accuracy/{dataset_name}_accuracy.png', bbox_inches='tight')
    plt.clf()
    
    return accuracy, timestamps, memory_usages



def loadLED(n):
    led_generator = synth.LEDDrift(seed=13, noise_percentage = 0.15, irrelevant_features=True, n_drift_features=3)
    led = []
    for data in led_generator.take(n):
        led.append(data)
    return led


def loadRBF(n):
    rbf_generator = synth.RandomRBFDrift(seed_model=13, seed_sample=13, n_classes=5, n_features=10, n_centroids=20, change_speed=0.87, n_drift_centroids=10)
    rbf = []
    lowest = 0
    for data in rbf_generator.take(n):
        x,y = data
        for feature in x:
            if x[feature] < lowest:
                lowest = x[feature]
        rbf.append(data)
    for x,_ in rbf:
        for feature in x:
            x[feature] += abs(lowest)
    return rbf


def loadAirlines():
    airlines_raw = arff.loadarff('../../datasets/airlines.arff')
    df_airlines = pd.DataFrame(airlines_raw[0])
    #df_airlines = df_airlines.head(1000)
    keys = df_airlines.keys()
    x_keys = keys[:7]
    y_key = keys[7]
    str_df_airlines = df_airlines.select_dtypes([object])
    str_df_airlines = str_df_airlines.stack().str.decode('utf-8').unstack()

    le = preprocessing.LabelEncoder()
    for col in str_df_airlines:
        le.fit(str_df_airlines[col])
        df_airlines[col] = le.transform(str_df_airlines[col])
    airlines = list()
    length = len(df_airlines) 
    for i in range(length):
        x = dict(df_airlines[x_keys].loc[i])
        y = int(df_airlines[y_key].loc[i]) == 1
        airlines.append((x, y))

    return airlines


def loadCovtype():
    covtype_raw = arff.loadarff('../../datasets/covtypeNorm.arff')
    df_covtype = pd.DataFrame(covtype_raw[0])
    keys = df_covtype.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]
    str_df_covtype = df_covtype.select_dtypes([object])
    str_df_covtype = str_df_covtype.stack().str.decode('utf-8').unstack()
    for col in str_df_covtype:
        n_different_values = len(set(str_df_covtype[col]))
        if n_different_values <= 2: # Check if column has more than two different values
            try:
                df_covtype[col] = list(map(bool, str_df_covtype[col].astype(int)))  # Convert column from string to boolean if value is "1" or "0"
            except ValueError:
                df_covtype[col] = str_df_covtype[col].astype(int) # Convert column from string to boolean given a condition
    covtype = list()
    length = len(df_covtype) 
    for i in range(length):
        x = dict(df_covtype[x_keys].loc[i])
        y = int(df_covtype[y_key].loc[i])
        covtype.append((x, y))
        
    return covtype


def loadAds():

    ads_raw = arff.loadarff('../../datasets/internet_ads.arff')
    df_ads = pd.DataFrame(ads_raw[0])
    keys = df_ads.keys()
    x_keys = keys[:len(keys)-1] # input keys
    y_key = keys[len(keys)-1] # class key
    str_df_ads = df_ads.select_dtypes([object]) # Select returns unprocessable type "object"
    str_df_ads = str_df_ads.stack().str.decode('utf-8').unstack() # Decode values into string
    for col in str_df_ads:
        n_different_values = len(set(str_df_ads[col]))
        if n_different_values <= 2: # Check if column has more than two different values
            try:
                df_ads[col] = list(map(bool, str_df_ads[col].astype(int)))  # Convert column from string to boolean if value is "1" or "0"
            except ValueError:
                df_ads[col] = str_df_ads[col] == "ad"  # Convert column from string to boolean given a condition
        else:
            raise ValueError('Field is not boolean')
    ads = list()
    length = len(df_ads) 
    for i in range(length):
        x = dict(df_ads[x_keys].loc[i])
        y = df_ads[y_key].loc[i]
        ads.append((x, y))
    
    return ads


def loadElec():
    elec_raw = arff.loadarff('../../datasets/elecNormNew.arff')
    df_elec = pd.DataFrame(elec_raw[0])
    keys = df_elec.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]
    str_df_elec = df_elec.select_dtypes([object])
    str_df_elec = str_df_elec.stack().str.decode('utf-8').unstack()
    for col in str_df_elec:
        try:
            df_elec[col] = str_df_elec[col].astype(int)
        except ValueError:
            df_elec[col] = str_df_elec[col] == "UP"
    elec = list()
    length = len(df_elec) 
    for i in range(length):
        x = dict(df_elec[x_keys].loc[i])
        y = df_elec[y_key].loc[i]
        elec.append((x, y))
    
    return elec


def loadKdd99():
    kdd99_raw = arff.loadarff('../../datasets/kdd99.arff')
    df_kdd99 = pd.DataFrame(kdd99_raw[0])
    keys = df_kdd99.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]

    str_df_kdd99 = df_kdd99.select_dtypes([object])
    str_df_kdd99 = str_df_kdd99.stack().str.decode('utf-8').unstack()
    le = preprocessing.LabelEncoder()
    for col in str_df_kdd99:
        n_different_values = len(set(str_df_kdd99[col]))
        if n_different_values <= 2: # Check if column has more than two different values
            try:
                df_kdd99[col] = list(map(bool, str_df_kdd99[col].astype(int)))  # Convert column from string to boolean if value is "1" or "0"
            except ValueError:
                le.fit(str_df_kdd99[col])
                df_kdd99[col] = le.transform(str_df_kdd99[col])
        else:
            le.fit(str_df_kdd99[col])
            df_kdd99[col] = le.transform(str_df_kdd99[col])

    kdd99 = list()
    length = len(df_kdd99) 
    for i in range(length):
        x = dict(df_kdd99[x_keys].loc[i])
        y = df_kdd99[y_key].loc[i]
        kdd99.append((x, y))

    return kdd99


def loadNomao():
    nomao_raw = arff.loadarff('../../datasets/nomao.arff')
    df_nomao = pd.DataFrame(nomao_raw[0])
    keys = df_nomao.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]

    str_df_nomao = df_nomao.select_dtypes([object])
    str_df_nomao = str_df_nomao.stack().str.decode('utf-8').unstack()
    for col in str_df_nomao:
        try:
            df_nomao[col] = str_df_nomao[col].astype(int)
        except ValueError:
            df_nomao[col] = str_df_nomao[col]
    nomao = list()
    length = len(df_nomao) 
    for i in range(length):
        x = dict(df_nomao[x_keys].loc[i])
        y = df_nomao[y_key].loc[i]
        nomao.append((x, y))

    return nomao


def loadSpam():
    spam_raw = arff.loadarff('../../datasets/spam_corpus.arff')
    df_spam = pd.DataFrame(spam_raw[0])
    keys = df_spam.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]

    str_df_spam = df_spam.select_dtypes([object])
    str_df_spam = str_df_spam.stack().str.decode('utf-8').unstack()
    for col in str_df_spam:
        n_different_values = len(set(str_df_spam[col]))
        if n_different_values <= 2: # Check if column has more than two different values
            try:
                df_spam[col] = list(map(bool, str_df_spam[col].astype(int)))  # Convert column from string to boolean if value is "1" or "0"
            except ValueError:
                df_spam[col] = str_df_spam[col]
        else:
            raise ValueError('Field is not boolean')
    spam = list()
    length = len(df_spam) 
    for i in range(length):
        x = dict(df_spam[x_keys].loc[i])
        y = df_spam[y_key].loc[i]
        spam.append((x, y))
        
    return spam



def loadDataset(dataset_name):
    data_raw, _ = arff.loadarff(f'../../datasets/{dataset_name}.arff')
    df_data = pd.DataFrame(data_raw)
    keys = df_data.keys()
    x_keys = keys[:len(keys)-1]
    y_key = keys[len(keys)-1]
    str_df_data = df_data.select_dtypes([object])
    str_df_data = str_df_data.stack().str.decode('utf-8').unstack()
    le = preprocessing.LabelEncoder()
    for col in str_df_data:
        if data_raw.dtype[col] == "S1":
            df_data[col] = list(map(bool, str_df_data[col].astype(int))) 
        else:
            le.fit(str_df_data[col])
            df_data[col] = le.transform(str_df_data[col])
    size = len(df_data) 
    for i in range(size):
        x = dict(df_data[x_keys].loc[i])
        y = df_data[y_key].loc[i]

    data = list()
    length = len(df_data) 
    for i in range(length):
        x = dict(df_data[x_keys].loc[i])
        y = df_data[y_key].loc[i]
        data.append((x, y))
    return data



def test_then_train_lite(data, model, model_name, dataset_name, file_name, accuracy, timestamps, memory_usages):
    #Test_then_train receiving accuracy, timestamps, memory_usages for use with split datasets

    acc_text = open(f"./results/{file_name}_accuracy.txt", "a")
    size = len(data) 
    step_size = ceil(size/10)
    lower_idx = 0
    upper_idx = step_size + size % 10

    print(f"Starting test-then-train for model {model_name} whith dataset {dataset_name}")
    for step in range(10):
        print(f"Step {step+1} ({(step+1)*10}%)")
        acc_text.write(f"\nStep {step+1} ({(step+1)*10}%)")

        training_data = data[lower_idx : upper_idx]

        total = 0
        correct = 0
        start_time = time.time()
        for x,y in training_data:
            y_pred = model.predict_one(x)
            total += 1
            if y_pred == y:
                correct += 1

            model.learn_one(x,y)
                
        end_time = time.time()


        timestamps[model_name] = timestamps.get(dataset_name, 0) + (end_time - start_time)
        memory_usages[model_name] = model._memory_usage
        if not dataset_name in accuracy:
            accuracy[dataset_name] = {}
        if not model_name in accuracy[dataset_name]:
            accuracy[dataset_name][model_name] = []

        accuracy[dataset_name][model_name].append(correct/total)
        acc_text.write(f"(MODEL {model_name}, {correct} correct out of {total} ({correct/total} accuracy)")
        lower_idx = upper_idx
        upper_idx = min(upper_idx + step_size, size)

    acc_text.write("\n\n")
    acc_text.close()

    time_text = open(f"./results/{file_name}_timestamps.txt", "a")
    time_text.write(f"{model_name}: {timestamps[model_name]}\n\n")
    time_text.close()
        
    mem_text = open(f"./results/{file_name}_memory_usage.txt", "a")

    mem_text.write(f"{model_name}: {memory_usages[model_name]}\n\n")
    mem_text.close()
        
    #print(accuracy)
    return accuracy, timestamps, memory_usages