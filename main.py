import tpot
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#load data
telescope = pd.read_csv('MAGIC Gamma Telescope Data.csv')

#clean data
telescope_shuffled = telescope.iloc[np.random.permutation(len(telescope))]#shuffle data
tele = telescope_shuffled.reset_index(drop=True) #reorder indices

#Store 2 classes
tele['Class'] = tele['Class'].map({'g':0, 'h':1})
tele_class = tele['Class'].values

#split training, validation and testing data
training_indices, validation_indices = training_indices, test_indices = train_test_split(tele.index,
                                    stratify=tele_class, train_size=0.75, test_size=0.25)

#Genetic Algorithm to find best ML model and hyperparameters
tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values, tele.loc[training_indices, 'Class'].values)

#score accuracy
tpot.score(tele.drop('Class', axis=1).loc[validation_indices].values, tele.loc[validation_indices, 'Class'].values)

#Export generated code
tpot.export('pipeline.py')