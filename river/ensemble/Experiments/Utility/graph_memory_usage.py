import re
import matplotlib.pyplot as plt

dataset_names = ["covtype"]
for dataset_name in dataset_names:
    memory_usages = {}
    filename = f"{dataset_name}_prueba_2_memory_usage"
    f = open(f'{filename}.txt', "r")
    for line in f:
        try:
            name = re.search(r'\w+', line).group(0)
            memory_usage = float(re.search(r'(?<=: )\d+(\.\d+)?', line).group(0))
            unit = re.search(r'[KMG]B', line).group(0)
            if unit == "KB":
                memory_usage *= 0.001
            elif unit == "GB":
                memory_usage *= 1000
            if name not in memory_usages:
                memory_usages[name] = []
            memory_usages[name] = memory_usage
        except:
            pass
    f.close()

    plt.clf()
    for model_name in memory_usages:
        plt.bar(model_name, memory_usages[model_name])
        plt.grid(visible=True)
        plt.title(dataset_name)
        plt.ylabel('max memory usage (MB)')
        #print(memory_usages[memory_usage])
    plt.savefig(f'{filename}.png', bbox_inches='tight')
