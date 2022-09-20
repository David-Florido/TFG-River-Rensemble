import os
from re import search
from math import ceil

def count_lines(filename):
    header_lines = 0
    content_lines = 0
    data = False
    with open(filename) as f:
        for lineno, line in enumerate(f):
            if not data:
                if(search("@data", line)):
                    data = True
                elif(search("@(attribute|relation)", line)):
                    header_lines += 1
            else:
                content_lines += 1
    print(f"header: {header_lines}")
    print(f"content: {content_lines}")
    
    lines_per_file = ceil(content_lines/10)
    print(f"lines_per_file: {lines_per_file}")
    return lines_per_file



def split_arff(dataset_name):
    lines_per_file = count_lines(f"{dataset_name}.arff")
    cont = 0
    smallfile = None
    written_data = 0
    data = False
    header = []
    os.makedirs(os.path.dirname(f"./{dataset_name}/"), exist_ok=True)
    with open(f'./{dataset_name}.arff') as bigfile:
        for lineno, line in enumerate(bigfile):
            if not data:
                if(search("@(attribute|relation)", line)):
                    header.append(line)
                elif(search("@data", line)):
                    data = True
                    header.append(line)
            else:
                if written_data % lines_per_file == 0:
                    if smallfile:
                        cont += 1
                        smallfile.close()
                    small_filename = f'{dataset_name}_{cont}.arff'
                    smallfile = open(f"./{dataset_name}/{small_filename}", "w")
                    for header_line in header:
                        smallfile.write(header_line)
                if(search("@(attribute|relation)", line)):
                    header.append(line)
                else:
                    smallfile.write(line)
                    written_data += 1
        if smallfile:
            smallfile.close()

split_arff("airlines")
split_arff("covtypeNorm")
split_arff("elecNormNew")
split_arff("internet_ads")
split_arff("kdd99")
split_arff("nomao")
split_arff("spam_corpus")