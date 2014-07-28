from __future__ import absolute_import

from imputers.celery import app

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR

from pymongo import MongoClient
mongo_client = MongoClient()
imputers = mongo_client.imputers

@app.task
## Data Slicer.
def data_slicer(file_path, bytes_max):
    data_file = open(file_path)
    colnames = data_file.readline()
    colnames = colnames.strip().split(',')
    lines = [line.strip().split(',') for line in data_file.readlines(bytes_max)]
    count = 1
    file_prefix = file_path[:-4]
    while lines != []:
        file_name = file_prefix + '_split' + '%.4d' % count +'.csv'
        print("Processing " + file_name)
        data = pd.DataFrame(lines, columns = colnames)
        data.to_csv(file_name, sep = ',', encoding = 'utf-8')
        lines = [line.strip().split(',') for line in data_file.readlines(bytes_max)]
        count += 1

@app.task
## Data Cleaner
def data_clean(input_path, output_path, C_cols, id_index = None, nan_rm = False, result = False):
    print('Reading ' + input_path)
    data = pd.DataFrame.from_csv(input_path)
    if id_index != None:
        print('Setting row names into columne %.20s.' % id_index)
        data = data.set_index(id_index)
    print("Factorizing categorical columns.")
    for col in C_cols:
        le = preprocessing.LabelEncoder()
        data[col].values.astype('a8')
        le.fit(data[col])
        data[col] = le.transform(data[col].values)
    if nan_rm:
        print('Removing NaN observations')
        for col in data.columns:
            data = data[np.isfinite(data[col])]
    print("Output " + output_path)
    data.to_csv(output_path)
    if result:
        return data


@app.task
## Imputer Trainer.
def imputers_train(data, col_classes):
    # Initializing an empty list to store the imputers
    imputers = []
    for col_ind in range(len(col_classes)):
        col = data.columns[col_ind]
        print("Processing the imputer of column " + col + ".")
        col_class = col_classes[col_ind]
        print("Column data type: " + col_class)
        Y = data[col]
        X = data.drop(col, 1)
        if col_class == 'categorical':
            rfc = RFC(n_estimators=20)
            imputer = rfc.fit(X, Y)
            imputers.append(imputer)
        elif col_class == 'numeric':
            rfr = RFR(n_estimators=20)
            imputer = rfr.fit(X, Y)
            imputers.append(imputer)
        else:
            pass
    return imputers
    