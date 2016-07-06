import gzip
import cPickle as pickle
import sys
import os
import subprocess
import os.path
import csv
from dircache import annotate
from scipy.misc import imresize

from scipy import misc
import numpy as np


def load_data(path="../../datasets/terrassa", download=True):
  previous_path=os.getcwd()
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)
  picklePath = path + "/terrassa.pickle"
  annotations_path = path + "/annotations.csv"
  # if file doesn't exist, create download and create the pickle, so next is fast
  if (os.path.isfile(picklePath)):
    print("Loading pickle previously created")
    f = open(picklePath, 'rb')
    data = pickle.load(f)
    f.close()
    os.chdir(previous_path)
    return data  # (X_train, y_train), (X_test, y_test)

  else:
    print("Getting & constructing the dataset. This can take a while and should only be executed once.")
    if (download):
      execute_script()
      print("Dataset downloaded.")

    print("Processing train.")
    X_train, y_train = load_folder(path + "/TerrassaBuildings900/train", annotations_path)
    print("Processing val.")
    X_val, y_val = load_folder(path + "/TerrassaBuildings900/val")
    #Add also real test if unknown data is wanted
    #print("Processing test.")
    #X_test, y_test = load_folder(path + "/test")

    print("Dumping pickle.")
    data = [(X_train, y_train), (X_val, y_val)]
    f = open(picklePath, "wb")
    pickle.dump(data, f)
    f.close()
    os.unlink("../../datasets/terrassa/terrassa900-test.zip")
    os.unlink("../../datasets/terrassa/terrassa900-trainval.zip")
    os.chdir(previous_path)
    return data


def load_data_without_unknown_class(path="../../datasets/terrassa", download=True):
  (X_train, y_train), (X_test, y_test) = load_data()
  X_train2 = list()
  y_train2 = list()
  X_test2 = list()
  y_test2 = list()

  for i in range(len(X_train)):
    if (y_train[i] == 3): continue
    X_train2.append(X_train[i])
    y_train2.append(y_train[i])

  for i in range(len(X_test)):
    if (y_test[i] == 3): continue
    X_test2.append(X_test[i])
    y_test2.append(y_test[i])

  X_train = np.asarray(X_train2)
  y_train = np.asarray(y_train2)
  X_test = np.asarray(X_test2)
  y_test = np.asarray(y_test2)
  data = [(X_train, y_train), (X_test, y_test)]
  return data

def execute_script():
  subprocess.call(['./terrassa_dataset.sh'])


def load_folder(path, annotationPath=None):
  annotations = dict()
  if (os.path.isfile(path + "/annotation.txt")):
    csv_reader = csv.reader(open(path + "/annotation.txt"), delimiter='\t')
    # skip first line
    next(csv_reader)
    annotations = dict(csv_reader)

  annotations_index = dict()
  curr_index = 0
  for value in sorted(set(annotations.values())):
    annotations_index[value] = curr_index
    curr_index = curr_index + 1
  if (annotationPath != None):
    f = open(annotationPath, 'wb')
    writer = csv.writer(f)
    for key, value in annotations_index.items():
      writer.writerow([key, value])
    f.close()
  x = []
  y = []
  path += "/images"
  for fn in os.listdir(path):
    if os.path.isfile(path + "/" + fn):
      if os.path.splitext(fn)[0] in annotations.keys():
        y.append(annotations_index[annotations[os.path.splitext(fn)[0]]])
      image = misc.imread(path + "/" + fn)
      image = imresize(image,[250,250])
      x.append(np.transpose(image, (2, 0, 1)))
  return np.asarray(x), np.asarray(y)


if __name__ == "__main__":
  data = load_data(download=False)
  print("Train: " + str(len(data[0][1])) + " Val: " + str(len(data[1][1])) )
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
