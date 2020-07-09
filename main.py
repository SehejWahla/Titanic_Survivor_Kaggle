import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import keras 
from keras import models
from keras import layers

from pathlib import Path
import os

# preprocessor performs data preprocessing on train and test data and returns the processed data as numpy array
from preprocessing import preprocessor

def ensemble_predictions(ml , train_np , labels , test_np):
    pred_by_vote = Voting_learning(ml , train_np , labels , test_np)
    pred_by_prob = Probability_learning(ml , train_np , labels , test_np)
    
    return pred_by_vote , pred_by_prob

def Voting_learning(ml , train_np , labels , test_np):
    for model in ml:
        model.fit(train_np , labels)
    Predictions_matrix = []
    
    for model in ml[:-1]:
        Predictions_matrix.append( model.predict(test_np) )
    
    nn_preds = (ml[-1].predict(test_np) > 0.5).astype(int)
    Predictions_matrix.append(nn_preds.squeeze())
    Predictions_matrix = np.array(Predictions_matrix)
    
    Votes = np.sum(Predictions_matrix , axis = 0)/len(ml)
    
    Votes = (Votes > 0.5).astype(int)
    
    return Votes

def Probability_learning(ml , train_np , labels , test_np):
    for model in ml:
        model.fit(train_np , labels)
    Predictions_matrix = []
    for model in ml[:-1]:
        Predictions_matrix.append( model.predict_proba(test_np) )
        
    nn_preds = (ml[-1].predict(test_np)).squeeze()

    Predictions_matrix = np.array(Predictions_matrix) 
    Predictions = np.sum( Predictions_matrix , axis = 0 ) /  ( len(ml) - 1 )

    probs = np.amax(  Predictions ,axis = 1)
    preds = np.argmax( Predictions, axis = 1 )

    preds_ens = []
    for x,y in zip(probs , preds):
        if y == 0:
            preds_ens.append(1.0 - x)
        else:
            preds_ens.append(x)
    preds_ens = np.array(preds_ens)

    pred_probs = preds_ens + nn_preds   
    pred_by_prob = (pred_probs > 0.5).astype(int)  
    return pred_by_prob

def create_file(arr , model_name):
    n = len(arr)
    id_s = np.arange(892 , 892+n)
    df = pd.DataFrame()
    df['PassengerId'] = id_s
    df['Survived'] = arr
    name = model_name+'.csv'
    df.to_csv(name, index = False)

df = pd.read_csv('data/train.csv')
Y = df['Survived'].values
df_test = pd.read_csv('data/test.csv')


#plotting all columns in train.csv , except 'PassengerId'
df.drop('PassengerId',axis=1).hist(bins=10,figsize=(9,7));

train_np , test_np =  preprocessor(df,df_test)

#Splitting train data into train and test for training models

X_train , X_test , Y_train , Y_test =   train_test_split( train_np , Y , 
                     random_state = 43 , 
                     stratify = Y , 
                     train_size = 0.8)

print('Train shape : ' , X_train.shape)
print('Test shape : ' , X_test.shape)


#testing performance of various models  using default hyperparameters
algos = [LogisticRegression() , DecisionTreeClassifier() , 
         RandomForestClassifier() , XGBClassifier() , SVC() , 
         GaussianNB()]

scores = []

for alg in algos:
  alg.fit(X_train , Y_train)
  Y_pred = alg.predict(X_test)
  scores.append(round(accuracy_score(Y_test , Y_pred) , 2))  

algo_names = ["LogisticRegression" , "DecisionTreeClassifier" , 
         "RandomForestClassifier" , "XGBClassifier" , "SVC" , 
         "GaussianNB"]

df_scores = pd.DataFrame()
df_scores['Model'] = np.array(algo_names)
df_scores['Score'] = np.array(scores)

df_scores.set_index('Model')


#Hyperparameter tuning XGBClassifier
model = XGBClassifier(silent=False, 
                      scale_pos_weight=2,
                      learning_rate=0.4,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=600, 
                      reg_alpha = 0.3,
                      max_depth=3, 
                      gamma=15)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
eval_metric = ["auc","error"]
get_ipython().run_line_magic('time', 'model.fit(X_train, Y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=500)')


#Building a neural network using Keras library and testing it's performance
bce = keras.losses.BinaryCrossentropy()
def create_model(num_cols , hidden_layers = [10]):
  model = models.Sequential()
  model.add( layers.Dense(hidden_layers[0] , activation = 'relu' , input_shape = (18,) ) )
  for n_cells in hidden_layers[1:]:
    model.add( layers.Dense(n_cells , activation='relu') )
  
  model.add( layers.Dense(1 , activation='sigmoid'))

  model.compile(
                optimizer = 'adam' , 
                loss = bce , 
                metrics = ['accuracy']
  )

  return model

nn_model = create_model(18 , [300,5])

nn_model.fit(X_train , Y_train , validation_data = (X_test , Y_test) , epochs = 5 , batch_size = 1 , shuffle = True)

svm = SVC( probability=True )
rf = RandomForestClassifier(n_estimators=200 , max_depth=5 , random_state = 43)
nb = GaussianNB()
lr = LogisticRegression()
xgb = XGBClassifier(silent=True, 
                      scale_pos_weight=2,
                      learning_rate=0.4,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=600, 
                      reg_alpha = 0.3,
                      max_depth=3, 
                      gamma=15)

model_list = [ svm , rf , nb , lr , xgb , nn_model ]

pred_by_vote , pred_by_prob =  ensemble_predictions(model_list , train_np , Y , test_np)

model_names = ['SVM' , 'RandomForest' , 'NaiveBayes' , 
               'LogisticRegression' , 'XGBoost' , 'NeuralNetwork' ,
               'Voting' , 'Probabilistic_voting']

#Saving predictions by various models

for model,name in zip( model_list[:-1] , model_names[:-3] ):
    model.fit(train_np , Y)
    preds = model.predict(test_np)
    create_file(preds , name)
    
nn_model.fit(train_np , Y)
preds = ( nn_model.predict(test_np) > 0.5 ).astype(int) 
create_file(preds , 'NeuralNetwork')

create_file(pred_by_vote , 'Pred_by_vote')
create_file(pred_by_prob , 'Pred_by_prob')
