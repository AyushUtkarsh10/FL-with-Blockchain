import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import numpy as np
import random
import re
import string
import requests
import gensim
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from sklearn.model_selection import train_test_split
import nltk
nltk.data.clear_cache()
from keras.models import model_from_json
import json
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

cols = list(pd.read_csv("cicddos2019_dataset.csv", nrows=1))
data = pd.read_csv("cicddos2019_dataset.csv", encoding='utf-8',
                   usecols =[i for i in cols if i != " Source IP"
                             and i != ' Destination IP' and i != 'Flow ID'
                             and i != 'SimillarHTTP' and i != 'Unnamed: 0'
                             and i != ' Inbound' and i != ' Fwd Header Length.1'])

label_counts = data['Label'].value_counts()

print(data.columns)

def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)

data = data.replace('Infinity','0')
data = data.replace(np.inf,0)
data['Flow Packets/s'] = pd.to_numeric(data['Flow Packets/s'])
data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(0)
data['Flow Bytes/s'] = pd.to_numeric(data['Flow Bytes/s'])

label_encoder = LabelEncoder()

data['Label'] = label_encoder.fit_transform(data['Label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['Class'] = label_encoder.fit_transform(data['Class'])
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

def get_data():
    cols = list(pd.read_csv("cicddos2019_dataset.csv", nrows=1))
    data = pd.read_csv("cicddos2019_dataset.csv", encoding='utf-8',
                       usecols=[i for i in cols if i != " Source IP"
                                and i != ' Destination IP' and i != 'Flow ID'
                                and i != 'SimillarHTTP' and i != 'Unnamed: 0'
                                and i != ' Inbound' and i != ' Fwd Header Length.1'])

    label_counts = data['Label'].value_counts()

    print(data.columns)

    def string2numeric_hash(text):
        import hashlib
        return int(hashlib.md5(text).hexdigest()[:8], 16)

    data = data.replace('Infinity', '0')
    data = data.replace(np.inf, 0)
    data['Flow Packets/s'] = pd.to_numeric(data['Flow Packets/s'])
    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(0)
    data['Flow Bytes/s'] = pd.to_numeric(data['Flow Bytes/s'])

    label_encoder = LabelEncoder()

    data['Label'] = label_encoder.fit_transform(data['Label'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    data['Class'] = label_encoder.fit_transform(data['Class'])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    y = pd.get_dummies(data["Class"])
    X = data.drop("Class", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=20)
    X_train = X_train.astype(np.float32)
    # X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # print(X_train_reshaped.shape[2])

    X_test = X_test.astype(np.float32)
    # X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    dataset = dict()
    dataset["train_text"] = X_train
    dataset["train_labels"] = y_train
    dataset["test_text"] = X_test
    dataset["test_labels"] = y_test
    return dataset

def save_data(dataset,name="sa.d"):
    '''
    Func to save mnist data in binary mode(its good to use binary mode)
    '''
    with open(name,"wb") as f:
        pickle.dump(dataset,f)

def load_data(name="sa.d"):
    '''
    Func to load mnist data in binary mode(for reading also binary mode is important)
    '''
    with open(name,"rb") as f:
        return pickle.load(f)

def get_dataset_details(dataset):
    '''
    Func to display information on data
    '''
    for k in dataset.keys():
        print(k,dataset[k].shape)

def split_dataset(dataset,split_count):
    '''
    Function to split dataset to federated data slices as per specified count so as to try federated learning
    '''
    datasets = []
    split_data_length = len(dataset["train_text"])//split_count
    for i in range(split_count):
        d = dict()
        d["test_text"] = dataset["test_text"][:]
        d["test_labels"] = dataset["test_labels"][:]
        d["train_text"] = dataset["train_text"][i*split_data_length:(i+1)*split_data_length]
        d["train_labels"] = dataset["train_labels"][i*split_data_length:(i+1)*split_data_length]
        datasets.append(d)
    return datasets


if __name__ == '__main__':
    save_data(get_data())
    dataset = load_data()
    get_dataset_details(dataset)
    for n,d in enumerate(split_dataset(dataset,5)):
        save_data(d,"federated_data_"+str(n)+".d")
        dk = load_data("federated_data_"+str(n)+".d")
        get_dataset_details(dk)
        print()
