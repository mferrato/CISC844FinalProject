import pandas as pd
import numpy as np
import data_manager as dm
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from statistics import mean
import models as m

clinical_path = 'clinical_data/TARGET_ALL_ClinicalData_Phase_I_20190325.xlsx'

## DATA CLEANING

# Creates dataset instance and loads the data
clinical_dataset = dm.Dataset()
clinical_dataset.load(clinical_path)

labels = clinical_dataset.get_labels()
labels = labels.copy()
df = clinical_dataset.get_dataset()

# List of features used
headers = ['Gender', 'Race', 'Ethnicity', 'Vital Status', 'First Event', 'Age at Diagnosis in Days', 'Year of Diagnosis', 'Year of Last Follow Up', 'WBC at Diagnosis', 'CNS Status at Diagnosis', 'MRD Day 8', 'MRD End Consolidation', 'Bone Marrow Site of Relapse', 'CNS Site of Relapse', 'Testes Site of Relapse', 'Other Site of Relapse', 'BMA Blasts Day 8', 'BMA Blasts Day 29', 'Down Syndrome', 'ALL Molecular Subtype']
df = pd.DataFrame(df, columns=headers)

# Target Identifiers to keep track of subjects
ids = df['TARGET USI']

# Removing non-relevant columns in the data frame
df.drop(columns=['TARGET USI', 'Vital Status', 'Comment '], inplace=True)


# Coverting BMA Blast features to categorical features
blast_8 = df['BMA Blasts Day 8'].apply(dm.blast_classification)
blast_29 = df['BMA Blasts Day 29'].apply(dm.blast_classification)
df['BMA Blasts Day 8'] = blast_8
df['BMA Blasts Day 29'] = blast_29

# Coverting MRD features to categorical features
mrd_8 = df["MRD Day 8"].apply(dm.mrd_classification)
mrd_end = df["MRD End Consolidation"].apply(dm.mrd_classification)
df['MRD End Consolidation'] = mrd_end
df['MRD Day 8'] = mrd_8

age = df['Age at Diagnosis in Days']//365
df['Age at Diagnosis in Days'] = age
df.rename(columns={'Age at Diagnosis in Days':'Age'}, inplace=True)


# Features that need one-hot encoding
string_attributes = ['Race', 'Ethnicity', 'CNS Status at Diagnosis', 'ALL Molecular Subtype']

# Applies the one-hot encoding
df = pd.get_dummies(df, columns=string_attributes)
df = df.fillna(0)

# Converts certain categorical values to numerical representation
ds = df['Down Syndrome'].apply(dm.categorical_string_to_number)
df['Down Syndrome'] = ds
ds = df['Testes Site of Relapse'].apply(dm.categorical_string_to_number)
df['Testes Site of Relapse'] = ds
ds = df['CNS Site of Relapse'].apply(dm.categorical_string_to_number)
df['CNS Site of Relapse'] = ds
ds = df['Bone Marrow Site of Relapse'].apply(dm.categorical_string_to_number)
df['Bone Marrow Site of Relapse'] = ds
ds = df['Other Site of Relapse'].apply(dm.categorical_string_to_number)
df['Other Site of Relapse'] = ds

# Changes gender representation to numerical
gdf = df['Gender'].apply(dm.gender_classification)
df['Gender'] = gdf

# Creates the dataset where only male subjects are present
male = df.loc[df['Gender'] == 0]

# Drops the label column to prevent data leakage
male_labels = male['First Event']
male = male.drop(columns='First Event')
df = df.drop(columns='First Event')
# Drops testes site of relapse as it does not apply for female subjects
df = df.drop(columns='Testes Site of Relapse')

## Regular dataset
print("Regular Dataset Results: ")
print(m.get_results(df, labels, 0))
## Male Only Dataset
print("Male Dataset Results: ")
print(m.get_results(male, male_labels, 0))
