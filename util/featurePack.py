import pandas as pd
import CKSAAP as ck
import PAAC as pa
import CKSAAGP as cg
import CTDD as cd
import CTriad as ctd
import Moran as ma
import SOCNumber as sn
import numpy as np
import torch

target_path = "../data3/Physcomitrella_heat_Pack.csv"


def getCKSAAP(seq):
    encodings = ck.CKSAAP(seq)
    code = np.array(encodings)
    return code


# Function to get CKSAAGP encoding
def getCKSAAGP(seq):
    encodings = cg.CKSAAGP(seq)
    code = np.array(encodings)
    return code


# Function to get PAAC encoding
def getPAAC(seq):
    encodings = pa.PAAC(seq)
    code = np.array(encodings)
    return code

def getCTDD(seq):
    encodings = cd.CTDD(seq)
    code = np.array(encodings)
    return code

def getCTriad(seq):
    encodings = ctd.CTriad(seq)
    code = np.array(encodings)
    return code

def getMoran(seq):
    encodings = ma.Moran(seq)
    code = np.array(encodings)
    return code

def getSOCNumber(seq):
    encodings = sn.SOCNumber(seq)
    code = np.array(encodings)
    return code


def make_tensor(path):
    data = pd.read_csv(path)
    sequences = data['sequence'].values
    labels = data['label'].values
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    for seq in sequences:
        temp1 = getPAAC(seq)
        temp2 = getCKSAAGP(seq)
        temp3 = getCKSAAP(seq)
        # temp4 = getCTDD(seq)
        # temp5 = getCTriad(seq)
        # temp6 = getMoran(seq)
        # temp7 = getSOCNumber(seq)
        features1.append(temp1)
        features2.append(temp2)
        features3.append(temp3)
        # features4.append(temp4)
        # features5.append(temp5)
        # features6.append(temp6)
        # features7.append(temp7)

    features1 = torch.FloatTensor(features1)
    features2 = torch.FloatTensor(features2)
    features3 = torch.FloatTensor(features3)
    # features4 = torch.FloatTensor(features4)
    # features5 = torch.FloatTensor(features5)
    # features6 = torch.FloatTensor(features6)
    # features7 = torch.FloatTensor(features7)
    features = torch.cat([features3, features1, features2], dim=1)
    labels = np.array(labels)
    labels=torch.tensor(labels)
    return features, labels


# source_pssm, source_label = make_tensor(source_path)
target_pssm, target_label = make_tensor(target_path)
# test_pssm, test_label = make_tensor(test_path)

# source_data = torch.cat([torch.tensor(source_pssm),source_label.reshape(source_label.shape[0],1)],dim=1)
target_data = torch.cat([torch.tensor(target_pssm),target_label.reshape(target_label.shape[0],1)],dim=1)

# df = pd.DataFrame(np.array(source_data))
df2 = pd.DataFrame(np.array(target_data))
# df.to_csv("../featurePack/sourceDataCKPA2.csv")
df2.to_csv("../data3/Physcomitrella_Pack.csv")