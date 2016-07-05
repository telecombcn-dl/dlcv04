
import gzip
import cPickle as pickle
import sys
import os
import subprocess
import os.path
import csv
from scipy import misc

def load_data(path="../../datasets/terrassa", download=True):

  #if file doesn't exist, create download and create the pickle, so next is fast
  if(os.path.isfile(path + "/terrassa.pickle")):
     f = open(path, 'rb')
     data = pickle.load(f)
     f.close()
     return data  # (X_train, y_train), (X_test, y_test)

  else:
    print("Getting the dataset. This can't take a while.")
    if(download):
      execute_script()
    print("Dataset downloaded.")

    X_train, y_train =  load_folder(path + "/TerrassaBuildings900/train")
    X_val, y_val = load_folder(path +  "/TerrassaBuildings900/val")
    X_test, y_test = load_folder(path +  "/test")

    f = open(path, "wb")
    data = (X_train, y_train),(X_val,y_val), (X_test,y_test)
    pickle.dump(data, f)
    f.close()
    return data


def execute_script():
  subprocess.call(['./terrassa_dataset.sh'])

def load_folder(path):
  if (os.path.isfile(path + "/annotation.txt")):
    csv_reader = csv.reader(open(path + "/annotation.txt"), delimiter='\t')
    #skip first line
    next(csv_reader)
    annotations = dict(csv_reader)
  x = []
  y = []
  path += "/images"
  for fn in os.listdir(path):
    if os.path.isfile(path + "/" + fn):
      x.append(annotations[os.path.splitext(fn)[0]])
      y.append(misc.imread(path + "/" + fn))
      print (fn)
  return x,y

if __name__ == "__main__":
  load_data(download=False)
#  (X_train, y_train), (X_test, y_test) = mnist.load_data()



