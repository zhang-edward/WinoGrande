import numpy as np
import pandas as pd 
import torch
import json
import random

# MaskedWiki (downsampled) download link: 
# https://ora.ox.ac.uk/objects/uuid:9b34602b-c982-4b49-b4f4-6555b5a82c3d/download_file?file_format=zip&safe_filename=MaskedWiki_downsampled.zip&type_of_work=Dataset

print ("Reading file...")
data = []
with open("WikiCREM_dev.txt") as f:
    i = 0
    example = []
    while True:
        line = f.readline()
        if line == "":
            break
        if line == "\n":
            data.append(example)
            example = []
        else:
            example.append(line.strip())
        i += 1
        print("Loaded {} examples".format(int((i+1)/5)), end="\r")
print("")

'''
{
"qID": "3QHITW7OYO7Q6B6ISU2UMJB84ZLAQE-2", 
"sentence": "Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.", 
"option1": "Ian", 
"option2": "Dennis",
"answer": "2"
}
'''

X = []
Y = []
for ex in data:
    if (len(ex) < 4):
        print("found bad example")
        continue
    sentence, token, options, label = ex
    option1, option2 = options.split(",")
    out_ex = {}
    out_ex['sentence'] = sentence.replace("[MASK]", "_")

    # option 1 is always correct
    if random.random()>0.5:
        out_ex['option1'] = option1
        out_ex['option2'] = option2
        answer = "1"
    else:
        out_ex['option1'] = option2
        out_ex['option2'] = option1
        answer = "2"

    out_ex['answer'] = answer
    X.append(out_ex)
    Y.append(answer)

with open('val_x.txt', 'w') as f:
    for x in X:
        f.write(json.dumps(x) + "\n")

with open('val_y.txt', 'w') as f:
    for y in Y:
        f.write(y + "\n")

# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = np.array([x.strip() for x in content])
# print ("Reshaping array...")
# data = np.reshape(content, (-1, 5))

# print("Loading into pandas DataFrame...")
# df = pd.DataFrame(data=data)
# df.to_csv(r'masked-wiki.csv', index = False)
# print(df.head())