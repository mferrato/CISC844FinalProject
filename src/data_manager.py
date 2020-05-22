import pandas as pd

data_directory = f'../dataset/'

class Dataset():
    def __init__(self):
        self.dataframe = None
        self.labels = None
    def load(self, path):

        # Reads dataset from excel file
        new_dataframe = pd.read_excel(data_directory + path)
        # Gets rid of the first two instances, which have no data
        new_dataframe = new_dataframe.iloc[2:]
        # Gets rid of all events not pertaining Relapse or Non-Relapse
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'Censored']
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'Death']
        new_dataframe = new_dataframe[new_dataframe['First Event'] != 'SMN']
        # Converts Label into binary (0 - None, 1 - Relapse)
        new_labels = new_dataframe['First Event'].apply(label_classification)
        new_dataframe['First Event'] = new_labels

        self.labels = new_dataframe['First Event']
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

# Converts MRD to categorical
def mrd_classification(x):
	if(x == 0):
		return 0 # No Risk
	elif(x >= 0 and x < 0.1):
		return 1 # Low Risk
	elif(x >= 0.1 and x < 1.0):
		return 2 # Medium Risk
	elif(x >= 1.0):
		return 3 # High Risk

# Converts Blast to categorical
def blast_classification(x):
	if(x <= 5):
		return 0 # Low Risk
	else:
		return 1 # High Risk

# Covert label to binary
def label_classification(x):
	if(x == 'None'):
		return 0 # None
	else:
		return 1 # Relapse

# Changes certain classification to a numeric representation
def categorical_string_to_number(x):
    if (x == 'No'):
        return 0
    elif (x == 'Yes'):
        return 1
    else:
        return 2

# Changes gender to a numerical representation
def gender_classification(x):
    if (x == 'Male'):
        return 0
    elif (x == 'Female'):
        return 1
