import numpy as np
import csv

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def load_csv_test_data(data_path, sub_sample=False, has_ID=False):

    raw = open(data_path, 'r').readlines()
    ids = np.zeros(len(raw))
    data = np.array(["" for _ in range(len(raw))], dtype='object')
    i = 0
    if has_ID:
        for line in raw:
            idd, dat = line.split(",", 1)
            ids[i] = int(idd)
            data[i] += dat
            i = i + 1
    else:
        for line in raw:
            data[i] += line
            i = i + 1


    return ids, data


def predict_labels(y_pred, treshold = 0):
    """Generates binary class predictions given a treshold and continuous predictions"""
    y_pred[np.where(y_pred <= treshold)] = -1
    y_pred[np.where(y_pred > treshold)] = 1
    
    return y_pred