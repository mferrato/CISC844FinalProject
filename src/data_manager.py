import pandas as pd

data_directory = f'../dataset/'



class Dataset():
    def __init__(self):
        self.dataframe = None
        self.labels = None
    def load(self, path):
        new_dataframe = pd.read_excel(data_directory + path)
        new_dataframe = new_dataframe.iloc[2:]
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'Censored'] 
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'Death']
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'SMN']
        new_labels = new_dataframe['First Event'].apply(label_classification) 
        new_dataframe['First Event'] = new_labels  
        self.labels = new_dataframe['First Event']
        #new_dataframe = new_dataframe.drop(columns='First Event')
        self.dataframe = new_dataframe
    def print(self):
        print(self.dataframe)
    def shape(self):
        return self.dataframe.shape
    def feature_list(self):
        return self.dataframe.columns
    def get_dataset(self):
        return self.dataframe
    def get_labels(self):
        return self.labels

def mrd_classification(x):
	if(x == 0):
		return 0
	elif(x >= 0 and x < 0.1):
		return 1
	elif(x >= 0.1 and x < 1.0):
		return 2
	elif(x >= 1.0):
		return 3

def blast_classification(x):
	if(x <= 5):
		return 0
	else:
		return 1

def label_classification(x):
	if(x == 'None'):
		return 0
	else:
		return 1