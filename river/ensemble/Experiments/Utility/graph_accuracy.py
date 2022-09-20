import re
import matplotlib.pyplot as plt

dataset_names = ["airlines", "LED"]
for dataset_name in dataset_names:
    accuracies = {}
    filename = f"{dataset_name}_prueba_2_accuracy"
    f = open(f'{filename}.txt', "r")
    for line in f:
        try:
            name = re.search(r'(?<=MODEL )\w+', line).group(0)
            accuracy = float(re.search(r'\d+(\.\d+)', line).group(0))
            if name not in accuracies:
                accuracies[name] = []
            accuracies[name].append(accuracy)
        except:
            pass
    f.close()

    for model_name in accuracies:
        #print(f'{model_name}: {accuracies[model_name]}')
        x = [10,20,30,40,50,60,70,80,90,100]
        plt.plot(x,accuracies[model_name], label=model_name)
        plt.xticks(x)
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.title(dataset_name)
        plt.xlabel('% of dataset used')
        plt.ylabel('accuracy')
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.grid(visible=True)
        plt.ylim([0, 1.1])
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.clf()