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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
import argparse
import random
import sys

clinical_path = f'clinical_data/TARGET_ALL_ClinicalData_Phase_I_20190325.xlsx'
clinical_dataset = dm.Dataset()
clinical_dataset.load(clinical_path)
#clinical_dataset.print()

#print(clinical_dataset.shape())
#print(clinical_dataset.feature_list())

labels = clinical_dataset.get_labels()
labels = labels.copy()
df = clinical_dataset.get_dataset()
headers = ['Gender', 'Race', 'Ethnicity', 'Vital Status', 'First Event', 'Age at Diagnosis in Days', 'Year of Diagnosis', 'Year of Last Follow Up', 'WBC at Diagnosis', 'CNS Status at Diagnosis', 'MRD Day 8', 'MRD End Consolidation', 'Bone Marrow Site of Relapse', 'CNS Site of Relapse', 'Testes Site of Relapse', 'Other Site of Relapse', 'BMA Blasts Day 8', 'BMA Blasts Day 29', 'Down Syndrome', 'ALL Molecular Subtype']
ids = df['TARGET USI']
df.drop(columns=['TARGET USI', 'Vital Status', 'Comment '], inplace=True)
df = pd.DataFrame(df, columns=headers)
#event_free = df['Event Free Survival Time in Days']//365
#overall_survival = df['Overall Survival Time in Days']//365
#df['Event Free Survival Time in Days'] = event_free
#df['Overall Survival Time in Days'] = overall_survival
blast_8 = df['BMA Blasts Day 8'].apply(dm.blast_classification)
blast_29 = df['BMA Blasts Day 29'].apply(dm.blast_classification)
df['BMA Blasts Day 8'] = blast_8
df['BMA Blasts Day 29'] = blast_29
mrd_8 = df["MRD Day 8"].apply(dm.mrd_classification)
df['MRD Day 8'] = mrd_8
mrd_end = df["MRD End Consolidation"].apply(dm.mrd_classification)
df['MRD End Consolidation'] = mrd_end
age = df['Age at Diagnosis in Days']//365
df['Age at Diagnosis in Days'] = age
df.rename(columns={'Age at Diagnosis in Days':'Age'}, inplace=True)
#print(df['Age'])
#print(df['MRD Day 8'])

string_attributes = ['Race', 'Ethnicity', 'CNS Status at Diagnosis', 'ALL Molecular Subtype'] 

df = pd.get_dummies(df, columns=string_attributes) 
df = df.fillna(0)

def down_syndrome(x):
    if (x == 'No'):
        return 0
    elif (x == 'Yes'):
        return 1
    else:
        return 2

def changing_gender(x):
    if (x == 'Male'):
        return 0
    elif (x == 'Female'):
        return 1


ds = df['Down Syndrome'].apply(down_syndrome) 
df['Down Syndrome'] = ds
ds = df['Testes Site of Relapse'].apply(down_syndrome) 
df['Testes Site of Relapse'] = ds
ds = df['CNS Site of Relapse'].apply(down_syndrome) 
df['CNS Site of Relapse'] = ds
ds = df['Bone Marrow Site of Relapse'].apply(down_syndrome) 
df['Bone Marrow Site of Relapse'] = ds
ds = df['Other Site of Relapse'].apply(down_syndrome) 
df['Other Site of Relapse'] = ds
gdf = df['Gender'].apply(changing_gender) 
df['Gender'] = gdf

print(df.columns)


male = df.loc[df['Gender'] == 0]
male_labels = male['First Event']
male = male.drop(columns='First Event')
df = df.drop(columns='First Event')
df = df.drop(columns='Testes Site of Relapse')
print(male['Gender'].value_counts())
print(male_labels.value_counts())

print(df.dtypes)
gdbParameters = {
'silent': [False],
'max_depth': [6, 10, 15, 20],
'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
'gamma': [0, 0.25, 0.5, 1.0],
'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
'n_estimators': [100]}

svmParameters = {
    'C':            np.arange( 1, 100+1, 1 ).tolist(),
    'kernel':       ['linear', 'rbf'],                   # precomputed,'poly', 'sigmoid'
    'degree':       np.arange( 0, 100+0, 1 ).tolist(),
    'gamma':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
    'coef0':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
    'shrinking':    [True],
    'probability':  [False],
    'tol':          np.arange( 0.001, 0.01+0.001, 0.001 ).tolist(),
    'cache_size':   [2000],
    'class_weight': [None],
    'verbose':      [False],
    'max_iter':     [-1],
    'random_state': [None],
    }

rfParameters = {
                   'bootstrap': [True, False],
                   'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [130, 180, 230]}

def trainingClassifiers(dataset, labels):
    print("Training")
    model_1 = RandomForestClassifier(random_state=0)
    model_2 = svm.SVC()
    model_3 = XGBClassifier(silence=1)
    randomized_search_1 = RandomizedSearchCV(model_1, rfParameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
    randomized_search_2 = RandomizedSearchCV(model_2, svmParameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
    randomized_search_3 = RandomizedSearchCV(model_3, gdbParameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                                scoring='roc_auc', refit=True, random_state=42)

    randomized_search_1.fit(dataset, labels)
    randomized_search_2.fit(dataset, labels)
    randomized_search_3.fit(dataset, labels)

    best_score_1 = randomized_search_1.best_score_
    best_params_1 = randomized_search_1.best_params_
    best_score_2 = randomized_search_2.best_score_
    best_params_2 = randomized_search_2.best_params_
    best_params_3 = randomized_search_3.best_params_

    return randomized_search_1, randomized_search_2, randomized_search_3, best_params_1, best_params_2, best_params_3

def validation(model, x_test, y_test, threshold, svm):

   # prediction = model.predict(x_test)
    if(svm):
        prediction = model.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, prediction)
        roc_score = roc_auc_score(y_test, prediction)
    #roc_score = auc(fpr, tpr)
    #predictions_1 = [round(value) for value in pred_1[i]]

        accuracy = accuracy_score(y_test, prediction)
        cm = confusion_matrix(y_test, prediction)

        sensitivity = cm[0][0]/ (cm[0][0] + cm[1][0])
        specificity = cm[1][1]/ (cm[1][1] + cm[0][1])
        return fpr, tpr, roc_score, accuracy, sensitivity, specificity
    else:
        prediction_proba = model.predict_proba(x_test)
        prediction_proba = prediction_proba[:, 1]

        prediction = []
        for pred in prediction_proba:
            if pred > threshold:
                prediction.append(1)
            else:
                prediction.append(0)
    # print("GROUND TRUTH: ")
    # print(y_test[0:10])
    #  print("PREDICTION: ")
    #  print(prediction[0:10])
    #  print("PREDICTION PROBA: ")
    # print(prediction_proba[0:10])
    #fpr, tpr, thresholds = roc_curve(y_test, prediction, pos_label=1)
        fpr, tpr, thresholds = roc_curve(y_test, prediction_proba)
        roc_score = roc_auc_score(y_test, prediction)
    #roc_score = auc(fpr, tpr)
    #predictions_1 = [round(value) for value in pred_1[i]]

        accuracy = accuracy_score(y_test, prediction)
        cm = confusion_matrix(y_test, prediction)

        sensitivity = cm[0][0]/ (cm[0][0] + cm[1][0])
        specificity = cm[1][1]/ (cm[1][1] + cm[0][1])
        return fpr, tpr, roc_score, accuracy, sensitivity, specificity

def calculateBestScore(model_array_1, x_test, y_test, threshold, svm):
        fpr_array = []
        tpr_array = []
        roc_array = []
        accuracy_array = []
        sensitivity_array = []
        specificity_array = []

        fpr, tpr, roc_score, accuracy, sensitivity, specificity = validation(model_array_1, x_test, y_test, threshold, svm)
        fpr_array.append(fpr)
        tpr_array.append(tpr)
        roc_array.append(roc_score)
        accuracy_array.append(accuracy)
        sensitivity_array.append(sensitivity)
        specificity_array.append(specificity)

        return fpr_array, tpr_array, roc_array, accuracy_array, sensitivity_array, specificity_array

def run():

    x_train, x_test, y_train, y_test = train_test_split(male, male_labels, test_size = .25, shuffle = True)
    rf, svm, gb, param1, param2, param3 = trainingClassifiers(x_train, y_train)


    fpr_rf, tpr_rf, roc_rf, acc_rf, sens_rf, spec_rf = calculateBestScore(rf, x_test, y_test, .6, 0) # RANDOM FOREST
    fpr_svm, tpr_svm, roc_svm, acc_svm, sens_svm, spec_svm = calculateBestScore(svm, x_test, y_test, .6, 1) # SVM
    fpr_gb, tpr_gb, roc_gb, acc_gb, sens_gb, spec_fb = calculateBestScore(gb, x_test, y_test, .6, 0) # GRADIENT BOOSTING

    #feature_importance_plot(x_train, y_train)

    random_forest = {
        "fpr": fpr_rf,
        "tpr": tpr_rf,
        "roc": roc_rf,
        "acc": acc_rf,
        "sens": sens_rf,
        "spec": spec_rf
    }

    svm = {
        "fpr": fpr_svm,
        "tpr": tpr_svm,
        "roc": roc_svm,
        "acc": acc_svm,
        "sens": sens_svm,
        "spec": spec_svm
    }

    gradient_boosting = {
        "fpr": fpr_gb,
        "tpr": tpr_gb,
        "roc": roc_gb,
        "acc": acc_gb,
        "sens": sens_gb,
        "spec": spec_fb
    }
    
    results = {
        "random_forest": random_forest,
        "svm": svm,
        "gradient_boosting": gradient_boosting
    }

    # FPR is an array of ints [0, 0, 0]   [1, 1, 1]  = [.5, .5 .5]
    # TPR is an array of ints

    # ROC is an int  
    # ACC is an int
    # SENS is an int
    # SPEC int

    #Random ForesT:
       # FPR [0, 0, 0]   [1, 1, 1]  = [.5, .5 .5]
        # TPR
        # ROC, ACC, SENS, SPEC

    # SVM

    # GDB
    return results

def feature_importance_plot(x_train, y_train):
    model = XGBClassifier(silence=1)
    model.fit(x_train, y_train)
    #important = model.get_score(importance_type='gain')
    #print(important)
    xgb.plot_importance(model)
    plt.show()

# results = []
# for i in range(0, 2):
#     results.append(run())
# print(results)

def getAverages():
    # Init Arrays to Store RF Values
    fpr_rf_avg = []
    tpr_rf_avg = []
    roc_rf_avg = []
    acc_rf_avg = []
    sens_rf_avg = []
    spec_rf_avg = []

    # Init Arrays to Store SVM Values
    fpr_svm_avg = []
    tpr_svm_avg = []
    roc_svm_avg = []
    acc_svm_avg = []
    sens_svm_avg = []
    spec_svm_avg = []

    # Init Arrays to Store GB Values
    fpr_gb_avg = []
    tpr_gb_avg = []
    roc_gb_avg = []
    acc_gb_avg = []
    sens_gb_avg = []
    spec_gb_avg = []

    # Simulate
    for i in range(0, 10):
        results = run()
        # RF Values
        fpr_rf_avg.append(results["random_forest"]["fpr"])
        tpr_rf_avg.append(results["random_forest"]["tpr"])
        roc_rf_avg.append(results["random_forest"]["roc"])
        acc_rf_avg.append(results["random_forest"]["acc"])
        sens_rf_avg.append(results["random_forest"]["sens"])
        spec_rf_avg.append(results["random_forest"]["spec"])

        # SVM Values
        fpr_svm_avg.append(results["svm"]["fpr"])
        tpr_svm_avg.append(results["svm"]["tpr"])
        roc_svm_avg.append(results["svm"]["roc"])
        acc_svm_avg.append(results["svm"]["acc"])
        sens_svm_avg.append(results["svm"]["sens"])
        spec_svm_avg.append(results["svm"]["spec"])

        # GB Values
        fpr_gb_avg.append(results["gradient_boosting"]["fpr"])
        tpr_gb_avg.append(results["gradient_boosting"]["tpr"])
        roc_gb_avg.append(results["gradient_boosting"]["roc"])
        acc_gb_avg.append(results["gradient_boosting"]["acc"])
        sens_gb_avg.append(results["gradient_boosting"]["sens"])
        spec_gb_avg.append(results["gradient_boosting"]["spec"])

    # Calculate and store averages
    fpr_tpr_index_rf = fpr_rf_avg.index(max(fpr_rf_avg, key=len))
    random_forest = {
        "fpr_avg": fpr_rf_avg[fpr_tpr_index_rf],
        "tpr_avg": tpr_rf_avg[fpr_tpr_index_rf],
        "roc_avg": np.mean(np.array(roc_rf_avg)),
        "acc_avg": np.mean(np.array(acc_rf_avg)),
        "sens_avg": np.mean(np.array(sens_rf_avg)),
        "spec_avg": np.mean(np.array(spec_rf_avg))
    }

    fpr_tpr_index_svm = fpr_svm_avg.index(max(fpr_svm_avg, key=len))
    svm = {
        "fpr_avg": fpr_rf_avg[fpr_tpr_index_svm],
        "tpr_avg": tpr_rf_avg[fpr_tpr_index_svm],
        "roc_avg": np.mean(np.array(roc_svm_avg)),
        "acc_avg": np.mean(np.array(acc_svm_avg)),
        "sens_avg": np.mean(np.array(sens_svm_avg)),
        "spec_avg": np.mean(np.array(spec_svm_avg))
    }

    fpr_tpr_index_gb = fpr_gb_avg.index(max(fpr_gb_avg, key=len))
    gradient_boosting = {
        "fpr_avg": fpr_rf_avg[fpr_tpr_index_gb],
        "tpr_avg": tpr_rf_avg[fpr_tpr_index_gb],
        "roc_avg": np.mean(np.array(roc_gb_avg)),
        "acc_avg": np.mean(np.array(acc_gb_avg)),
        "sens_avg": np.mean(np.array(sens_gb_avg)),
        "spec_avg": np.mean(np.array(spec_gb_avg))
    }

    # Return results as a dict object
    results = {
        "random_forest": random_forest,
        "svm": svm,
        "gradient_boosting": gradient_boosting
    }
    return results

print(getAverages())
