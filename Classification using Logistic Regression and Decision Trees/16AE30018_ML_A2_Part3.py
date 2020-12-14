import numpy as np
import pandas as pd
from math import log2 as log
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

eps=10**-8

Class= 'quality'

def find_entropy(df):
    E = 0
    values = df[Class].unique()
    for val in values:
        frac = df[Class].value_counts()[val]/len(df[Class])
        E -= frac*np.log(frac)
    return E

def find_entropy_attribute(df,attribute):
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    E2 = 0
    for variable in variables:
        E = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            frac = num/(den+eps)
            E -= frac*log(frac+eps)
        frac2 = den/len(df)
        E2 -= frac2*E
    return abs(E2)

def find_highest(df):
    Info_Gain = []
    for col in df.keys()[:-1]:
        Info_Gain.append(find_entropy(df)-find_entropy_attribute(df,col))
    return df.keys()[:-1][np.argmax(Info_Gain)]

def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)

def buildTree(df,tree=None):
    node = find_highest(df)
    if tree is None:                    
        tree={}
    else:
        tree[node] = {}
    
    attValue = np.unique(df[node])
    
   #We make loop to construct a tree by calling this function recursively. 
   #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            flag=True
            for i in counts:
                if(i<10):
                    flag=False
            if(flag):
                tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree



def main():
    
    df= pd.read_csv("data/dataB.csv")

    clf=DecisionTreeClassifier(criterion= 'entropy', min_samples_split=10, random_state=5)
    X=df.drop(['quality'], axis=1).values
    y= df['quality'].values
    y = label_binarize(y, classes=[0, 1, 2])
    kf = KFold(n_splits=3, random_state=2, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        dt= clf.fit(X_train, y_train)
        y_sk= dt.predict(X_test)
        
        sk_accuracy=[]
        sk_recall=[]
        sk_precision=[]
    
        accuracy = accuracy_score(y_test, y_sk)
        precision = precision_score(y_test, y_sk)
        recall = recall_score(y_test, y_sk)
        sk_accuracy.append(accuracy)
        sk_precision.append(precision)
        sk_recall.append(recall)
        
    print("Mean Accuracy")
    print("SK Learn package:", np.mean(np.array(sk_accuracy)))
    print("Mean Precision")
    print("SK Learn package:", np.mean(np.array(sk_precision)))
    print("Mean recall")
    print("SK Learn package:", np.mean(np.array(sk_recall)))
    
if __name__=="__main__":
    main()