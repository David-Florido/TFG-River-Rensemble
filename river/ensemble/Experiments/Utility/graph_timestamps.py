import re
import matplotlib.pyplot as plt

dataset_names = ["airlines", "covtype", "LED", "RBF"]
for dataset_name in dataset_names:
    timestamps = {}
    filename = f"{dataset_name}_prueba_2_timestamps"
    f = open(f'{filename}.txt', "r")
    for line in f:
        try:
            name = re.search(r'\w+', line).group(0)
            timestamp = float(re.search(r'(?<=: )\d+(\.\d+)?', line).group(0))
            if name not in timestamps:
                timestamps[name] = []
            timestamps[name] = timestamp
        except:
            pass
    f.close()

    plt.clf()
    for model_name in timestamps:
        plt.bar(model_name, timestamps[model_name])
        plt.grid(visible=True)
        plt.title(dataset_name)
        plt.ylabel('time (s)')
        #print(timestamps[timestamp])
    plt.savefig(f'{filename}.png', bbox_inches='tight')
