
import gzip
import cPickle as pickle
import sys
import os
import subprocess
import os.path
import csv

def load_data():
  path = "../datasets/terrassa/terrassa.pkl"

  #if file doesn't exist, create download and create the pickle, so next is fast
  if(os.path.isfile(path)):
     f = open(path, 'rb')
     data = pickle.load(f)
     f.close()
     return data  # (X_train, y_train), (X_test, y_test)

  else:
    print("Getting the dataset. This can't take a while.")
    execute_script()
    print("Dataset downloaded.")

    X_train, y_train =  load_folder("../datasets/terrassa/TerrassaBuildings900/train/")
    X_train, y_train = load_folder("../datasets/terrassa/TerrassaBuildings900/val/")
    X_test, y_test = load_folder("../datasets/terrassa/TerrassaBuildings900/val/")

    f = open(path, "wb")
    data = [(X_train, y_train),(X_test,y_test)]
    pickle.dump(data, f)
    f.close()
    return data


def execute_script():
  FNULL = open(os.devnull, 'w')
  subprocess.call('./terrassa_dataset.sh')

def load_folder(path):
  path += "/images"
  if (os.path.isfile(path + "/annotation.txt")):
    csv_reader = csv.reader(open('text.txt', 'rb'), delimiter='\t')
    #skip first line
    next(csv_reader)
    annotations = list(csv_reader)




if __name__ == "__main__":
  load_data()
#  (X_train, y_train), (X_test, y_test) = mnist.load_data()


