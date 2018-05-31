# This snippet has to be run in Python 2.7, since load_data has been written
# using Python 2.7 and cPickle.

from mnist_loader import load_data
import pandas as pd
import numpy as np


def format_output(df, number):
    dfc = df.copy()
    dfc.loc[dfc["ClassLabel"] != number, "ClassLabel"] = 10
    dfc.loc[dfc["ClassLabel"] == number, "ClassLabel"] = 1
    dfc.loc[dfc["ClassLabel"] == 10, "ClassLabel"] = 0
    return dfc


tr_d, v_d, te_d = load_data()

# tr_d: Training set:
#  - tr_d[0] = List of 50000 elements (Images)
#       - tr_d[0][j] = List of 784 float numbers (Greyscale value of each pixel)
#       - tr_d[0][j][k] = Single pixel
#  - tr_d[1] = List of 50000 integer values (Class label 0-9)

col_names = []

for i in range(0, 784):
    col_names.append("P"+str(i))

col_names.append("ClassLabel")

images = tr_d[0].tolist()
labels = tr_d[1].tolist()

for image, label in zip(images, labels):
    image.append(label)

dataset = pd.DataFrame(images, columns=col_names)

for i in range(0, 10):
    df = dataset.copy()
    df = format_output(dataset, i)
    df.to_csv(r"mnist_datasets2\training_"+str(i) + ".csv",
              sep=',', header=True, columns=col_names, mode='w')
#dataset = format_output(dataset, 1)


# print dataset["ClassLabel"]
