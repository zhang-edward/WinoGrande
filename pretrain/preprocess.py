import numpy as np
import pandas as pd 
import torch

# MaskedWiki (downsampled) download link: 
# https://ora.ox.ac.uk/objects/uuid:9b34602b-c982-4b49-b4f4-6555b5a82c3d/download_file?file_format=zip&safe_filename=MaskedWiki_downsampled.zip&type_of_work=Dataset

print ("Reading file...")
data = []
with open("MaskedWiki_sample.txt") as f:
    i = 0
    example = []
    while i < 100000:
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
X = []
y = []
for ex in data:
    if (len(ex) < 4):
        print("found bad example")
        continue
    sentence, token, options, label = ex
    option1, option2 = options.split(",")
    X.append(sentence.replace(token, option1 + " [SEP] "))
    X.append(sentence.replace(token, option2 + " [SEP] "))
    if (label == option1):
        y.append([0.0, 1.0])
        y.append([1.0, 0.0])
    else:
        y.append([1.0, 0.0])
        y.append([0.0, 1.0])

with open('masked-wiki_X.txt', 'w') as f:
    for x in X:
        f.write(x + "\n")

y = torch.tensor(y)
torch.save(y, 'masked-wiki_y.pt')

# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = np.array([x.strip() for x in content])
# print ("Reshaping array...")
# data = np.reshape(content, (-1, 5))

# print("Loading into pandas DataFrame...")
# df = pd.DataFrame(data=data)
# df.to_csv(r'masked-wiki.csv', index = False)
# print(df.head())