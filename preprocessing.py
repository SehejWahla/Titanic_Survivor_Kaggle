import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocessor(df , df_test):    
    df['Name'] = df['Name'].apply( lambda x : x.split()[1])
    df_test['Name'] = df_test['Name'].apply( lambda x : x.split()[1] )
    df['Ticket'] = df['Ticket'].apply(lambda x : x.split()[-1])
    df_test['Ticket'] = df_test['Ticket'].apply(lambda x : x.split()[-1])
    df = df.drop(['Cabin'] , axis =1 )
    df_test = df_test.drop(['Cabin'] , axis = 1)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df_test['Sex'] = LabelEncoder().fit_transform(df_test['Sex'])
    l = ['Mr.' , 'Miss.' , 'Mrs.']
    df['Title'] =   df['Name'].apply( lambda x : 'Other' if ( x not in l ) else x )
    df_test['Title'] = df_test['Name'].apply( lambda x : 'Other' if ( x not in l ) else x )
    df = df.drop(['Name'] , axis = 1)
    df_test = df_test.drop(['Name'] , axis = 1)
    df['Age'] =   df.groupby(['Title','Sex'])['Age'].transform(
        lambda grp : grp.fillna(np.mean(grp))
    )
    df_test['Age'] = df_test.groupby(['Title','Sex'])['Age'].transform(
        lambda grp : grp.fillna(np.mean(grp))
    )
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    df_test['Title'] = LabelEncoder().fit_transform(df_test['Title'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df_test['Embarked'] = LabelEncoder().fit_transform(df_test['Embarked'])
    df['family_size']  =  df['SibSp'] + df['Parch']
    df_test['family_size']  =  df_test['SibSp'] + df_test['Parch']
    to_one_encode = ['Pclass' , 'Sex' , 'Embarked' , 'Title']
    for col in to_one_encode:
      df[col] = df[col].astype(object)
    for col in to_one_encode:
      df_test[col] = df_test[col].astype(object)
    for t in range(len(df['Ticket'])):
      if df.at[t,'Ticket'] == 'LINE':
        df.at[t,'Ticket'] = 0

    df['Ticket'] = df['Ticket'].astype(int)
    df_test['Ticket'] = df_test['Ticket'].astype(int)
    df_test['Fare'] =   df_test['Fare'].fillna(df['Fare'].mean())
    df = df.drop('PassengerId' , axis = 1)
    df_test = df_test.drop('PassengerId' , axis = 1)
    df = pd.get_dummies(df)
    df_test = pd.get_dummies(df_test)
    to_std =   ['Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'family_size']
    scaler = StandardScaler()
    numpy_train = df[to_std].values
    numpy_test = df_test[to_std].values
    scaler.fit(numpy_train)
    train_scaled = scaler.transform(numpy_train)
    test_scaled = scaler.transform(numpy_test)
    categ_train_numpy =  df.drop(to_std + ['Survived'] , axis = 1).values
    categ_test_numpy = df_test.drop(to_std , axis = 1).values
    Y = df['Survived'].values
    assert categ_test_numpy.shape[1] == categ_train_numpy.shape[1]
    train_np =   np.concatenate( ( train_scaled , categ_train_numpy ) , axis = 1 )
    test_np = np.concatenate( ( test_scaled , categ_test_numpy ) , axis = 1 )
    
    return train_np, test_np