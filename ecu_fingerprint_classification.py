# -*- coding: utf-8 -*-
"""ecu_fingerprint_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Pt0_Ad2MxJTRRJ9Qg9Yf4DiyJtSgvmPx

# In-Vehicle Security using Pattern Recognition Techniques
---
*ECE 5831 - 08/17/2021 - Kunaal Verma*

The goal of this project is to train a machine learning model to recognize unique signatures of each ECU in order to identify when intrusive messages are being fed to the network.

The dataset for this project consists of several time-series recordings of clock pulses produced by several ECUs on an in-vehicle CAN High network.

The test network has 8 trusted ECUs, with 30 records of 600 samples of the clock signal for each ECU.

# I. Initialization
"""

### I. Initialization ###
# print('--- Intialize ---')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sn
import sklearn.metrics as sm

# from google.colab import drive
# drive.mount('/content/drive')

"""# II. File Pre-conditioning

"""

### II. File Pre-conditioning ###
print('\n--- Pre-condition Files ---\n')

# Dataset path
# datapath = 'drive/MyDrive/School/UMich Dearborn/SS21/ECE 5831/Project/Dataset'
datapath = './Dataset'

print('Path to Dataset:')
print(datapath)

# List of files in directory, append to list recursively
file_list = []
for path, folders, files in os.walk(datapath):
    for file in files:
        file_list.append(os.path.join(path, file))

# Sort file list by record number, but use a copy that contains leading zeros
file_list0 = []
for filename in file_list:
    m = re.search('_[0-9]\.', filename)
    if m:
      found = m.group(0)
      filename = re.sub('_[0-9]\.', found[0] + '0' + found[1:-1] + '.', filename)
    file_list0.append(filename)

file_list1 = dict(zip(file_list0,file_list))
file_list2 = dict(sorted(file_list1.items(), key = lambda x:x[0]))

# Produce sorted file lists
file_list  = list(file_list2.values())  # Original list, properly sorted
file_list0 = list(file_list2.keys())    # Modified list with leading zeros, sorted

print('\nAbsolute Paths to Records (Subset):')
for filename in file_list[0:10]:
  print(filename)

"""# III. Feature Extraction


"""

### III. Feature Extraction ###
print('\n--- Extract Features ---\n')

# !cp '/content/drive/MyDrive/School/UMich Dearborn/SS21/ECE 5831/Project/github/ecu_fingerprint_lib.py' .
from ecu_fingerprint_lib import Record

pkl_path = './ecu_fingerprint.pkl'

if os.path.exists(pkl_path):
  
  print('Extracting Feature Data from previous run')
  df = pd.read_pickle(pkl_path)

else:

  # Dict instantiation
  d = {}

  # Iteratively build data dictionary
  for i in np.arange(len(file_list0)):

    filepath  = file_list[i]
    filepath0 = file_list0[i]

    print(filepath)

    # Extract folder name of current record
    folder = os.path.basename(os.path.dirname(filepath0))
    filename = os.path.basename(filepath0)

    # Extract record identifiers
    id, pl, pm, did = folder.split('_')
    filename = re.split(r'_|\.',filename)

    # Open File
    with open(filepath) as file_name:
      array = np.loadtxt(file_name, delimiter=",")
  
    # Extract Features
    r = Record(array)
  
    # Add Features and File Attributes to Dict
    if i == 0:
      #   File Metadata
      d['Filepath']   = []
      d['CAN_Id']     = []
      d['CAN_PhyLen'] = []
      d['CAN_PhyMat'] = []
      d['CAN_RecId']  = []
      #   Feature Data
      for feature_name in r.headers:
        d[feature_name]   = []

    # Build data table
    for k in np.arange(r.total_rec):
      for j in np.arange(len(r.headers)):
        d[r.headers[j]].append(r.features[j][k])
      d['Filepath'].append(filepath)
      d['CAN_Id'].append(id)
      d['CAN_PhyLen'].append(pl)
      d['CAN_PhyMat'].append(pm)
      d['CAN_RecId'].append(filename[-2])

  df = pd.DataFrame.from_dict(d)
  df.to_pickle(pkl_path)

print('\nDataFrame Object:\n')
print(df.info())

"""# IV. Prepare Training and Test Datasets"""

### IV. Prepare Training and Test Datasets ###
print('\n--- Training and Test Datasets ---\n')

# Attribute and Label Datasets

## Full, Spectral, Contrl, Dom, Rec                                                                     Acc %
# X = df.iloc[:,5:]                                                               # Full feature set    98.76%
# X = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23,24,32,33,34,35,36,37,38,39]] # Spectral Only       97.26%
# X = df.iloc[:,[5,6,7,8,9,10,11,25,26,27,28,29,30,31]]                           # Control Only        97.17%
# X = df.iloc[:, 5:24]                                                            # Dominant Only       98.14%
# X = df.iloc[:,25:39]                                                            # Recessive Only      96.73%

## Spectral Features (sorted by Accuracy)
# X = df.iloc[:,[22,37]]                                                          # SNR                 95.23%
# X = df.iloc[:,[23,38]]                                                          # Mean Freq           94.96%
# X = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,32,33,34,35,36]]                   # Spectral Density    92.93%
# X = df.iloc[:,[24,39]]                                                          # Median Freq         91.34%

## Spectral Features (added by rank)
# X = df.iloc[:,[22,37]]                                                          # SNR                 95.23% [ ]
# X = df.iloc[:,[22,23,37,38]]                                                    #  + Mean Freq        97.35% [+] <<<
# X = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38]]       #  + Spectral Density 97.26% [-]
# X = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38]]       #  + Median Freq      97.26% [-]

## Control Features (sorted by Accuracy)
# X = df.iloc[:,[ 8,28]]                                                          # Steady State Value  95.23%
# X = df.iloc[:,[ 9,29]]                                                          # Steady State Error  95.05%
# X = df.iloc[:,[ 6,26]]                                                          # Percent Overshoot   91.52%
# X = df.iloc[:,[ 7,27]]                                                          # Settling Time       91.52%
# X = df.iloc[:,[10,30]]                                                          # Rise Time           83.48%
# X = df.iloc[:,[11,31]]                                                          # Delay Time          81.89%
# X = df.iloc[:,[ 5,25]]                                                          # Peak Time           80.12%

## Control Features (added by rank)
# X = df.iloc[:,[8,28]]                                                           # SSV                 95.23% [ ]
# X = df.iloc[:,[8,9,28,29]]                                                      #  + SSE              97.70% [+]
# X = df.iloc[:,[6,8,9,26,28,29]]                                                 #  + %OS              98.32% [+] <<<
# X = df.iloc[:,[6,7,8,9,26,27,28,29]]                                            #  + Ts               98.32% [=]
# X = df.iloc[:,[6,7,8,9,10,26,27,28,29,30]]                                      #  + Tr               97.26  [-]
# X = df.iloc[:,[6,7,8,9,10,11,26,27,28,29,30,31]]                                #  + Td               97.17% [-]
# X = df.iloc[:,[ 5,25]]                                                          #  + Tp               97.17% [=]

## All Features (added by rank)
# X = df.iloc[:,[22,37]]                                                                   # SNR                95.23% [ ]
# X = df.iloc[:,[8,22,28,37]]                                                              #  + SSV             97.53% [+]
# X = df.iloc[:,[8,9,22,28,29,37]]                                                         #  + SSE             97.97% [+]
# X = df.iloc[:,[8,9,22,23,28,29,37,38]]                                                   #  + Mean Freq.      98.59% [+]
# X = df.iloc[:,[6,8,9,22,23,26,28,29,37,38]]                                              #  + %OS             98.94% [+] <<<
# X = df.iloc[:,[6,7,8,9,22,23,26,27,28,29,37,38]]                                         #  + Ts              98.85% [-]
# X = df.iloc[:,[6,7,8,9,22,23,24,26,27,28,29,37,38,39]]                                   #  + Med. Freq.      98.67% [-]
# X = df.iloc[:,[6,8,9,12,13,14,15,16,17,18,19,20,21,22,23,26,28,29,32,33,34,35,36,37,38]] #  + SD              98.32% [-]
# X = df.iloc[:,[10,30]]                                                                   #  + Tr              ...
# X = df.iloc[:,[11,31]]                                                                   #  + Td              ...
# X = df.iloc[:,[ 5,25]]                                                                   #  + Tp              ...

## Final Feature Set: (Control) SSV, SSE, %OS (Spectral) SNR, Mean Freq.
X = df.iloc[:,[6,8,9,22,23,26,28,29,37,38]]                                     # 99.12%

y = df.iloc[:,1]

# Change Labels from Categorical to Numerical values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(y)

# Train-Test Split (70/30)

from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)    # Apples to Apples Debugging
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print('Training Input:\n')
print('    %s: %s' %(X_train.dtype, X_train.shape))
print('\nTraining Output:\n')
print('    %s: %s' %(Y_train.dtype, Y_train.shape))
print('\nTest Input:\n')
print('    %s: %s' %(X_test.dtype, X_test.shape))
print('\nTest Output:\n')
print('    %s: %s' %(Y_test.dtype, Y_test.shape))

"""# V. Train Neural Network"""

### V. Train Neural Network ###
print('\n--- Neural Network Training ---\n')

from sklearn.neural_network import MLPClassifier

hls = 250 
mit = 500

mlp = MLPClassifier(hidden_layer_sizes=([hls,]), max_iter=mit)
mlp.fit(X_train, Y_train.ravel())
Y_pred = mlp.predict(X_test)
print('Hidden Layer Nodes:    ',len(mlp.coefs_[1]))
print('Neural Network Solver: ',mlp.solver)

"""# VI. Produce Confusion Matrix"""

### VI. Produce Confusion Matrix ###
print('\n--- Confusion Matrix ---\n')

cm = sm.confusion_matrix(Y_test, Y_pred)
tr = cm.size       # Total Records
# np.sum(cm, axis=0) # Column Sum     (Horizontal Result)
# np.sum(cm, axis=1) # Row Sum        (Vertical Result)
# np.trace(cm)       # Diagonal Sum   (Total Result)

# df_cm = pd.DataFrame(cm, y.unique(), y.unique())
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
# plt.xlabel("Prediction", fontsize = 12)
# plt.ylabel("Actual", fontsize = 12)

# plt.show()

# print('')
print(cm)

"""# VII. Evaluate Peformance Metrics"""

# Commented out IPython magic to ensure Python compatibility.
### VII. Evaluate Performance Metrics ###
print('\n--- Performance Metrics ---\n')

def perf_metrics(cm, report=False):

  cm_acc = []
  cm_pre = []
  cm_rec = []
  cm_f1s = []
  cm_err = []

  if report:
    print('[ ECU_Id | Accuracy | Precision | Recall | F1 Score | Error ]')
    print('=========|==========|===========|========|==========|========')

  for i in np.arange(len(cm)):
    cm_working = cm.copy()
    idx = np.concatenate((np.arange(0,i),np.arange(-len(cm)+i+1,0)))

    tp = cm_working[i,i]
    fn = cm_working[i,idx]; fn = np.sum(fn)
    fp = cm_working[idx,i]; fp = np.sum(fp)

    cm_working[i,i] = 0
    cm_working[i,idx] = 0
    cm_working[idx,i] = 0

    tn = np.sum(cm_working)

    acc = (tp + tn)/(tp + tn + fp + fn)
    err = 1-acc
    pre = tp/(tp + fp)
    rec = tp/(tp + fn)
    f1s = 2*(pre*rec)/(pre+rec)

    cm_acc.append(acc)
    cm_pre.append(pre)
    cm_rec.append(rec)
    cm_f1s.append(f1s)
    cm_err.append(err)

    if report:
      print('[%7s | %8.3f | %9.3f | %6.3f | %8.3f | %5.3f ]' \
	% (y.unique()[i], acc, pre, rec, f1s, err))

  return cm_acc, cm_pre, cm_rec, cm_f1s, cm_err

acc, pre, rec, f1s, err = perf_metrics(cm, True)

# Plot results

bw = 0.15   # Box plot bar width

p1 = np.arange(len(acc))    # Category x-axis positions
p2 = [x + bw for x in p1]
p3 = [x + bw for x in p2]
p4 = [x + bw for x in p3]
p5 = [x + bw for x in p4]

print('') 
plt.bar(p1, acc, width=bw)
plt.bar(p2, pre, width=bw)
plt.bar(p3, rec, width=bw)
plt.bar(p4, f1s, width=bw)
plt.bar(p5, err, width=bw)
 
plt.xticks([p + bw for p in range(len(acc))], y.unique(), fontsize = 10)
plt.legend(['Accuracy','Precision','Recall','F1 Score','Error'], fontsize = 10, bbox_to_anchor = (1, 0.67))

print(np.average(acc))
