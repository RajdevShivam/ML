import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    l= (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return l

def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))

def predict(X, w, threshold=0.5):
    return predict_probs(X, w) >= threshold

def logistic_regression(X, y, num_steps=100000, learning_rate=0.01):
    
    w = np.zeros(X.shape[1])
    
    for i in range(num_steps):
        z = np.dot(X, w)
        h = sigmoid(z)

        # Update weights with gradient
        error = y - h
        gradient = np.dot(X.T, error)
        w += learning_rate * gradient
        
    return w

def main():

    dataA= pd.read_csv("data/dataA.csv")
    X= dataA.drop(["quality"], axis=1).values
    y= dataA["quality"].values
    
    own_accuracy=[]
    sk_accuracy=[]
    own_recall=[]
    sk_recall=[]
    own_precision=[]
    sk_precision=[]
    
    LR= LogisticRegression(penalty= 'none', solver= 'saga')
    kf = KFold(n_splits=3, random_state=2, shuffle=True) #Used for 3 fold cross-validation
    
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        w= logistic_regression(X_train, y_train)
        y_own= predict(X_test, w, 0.5)
        
        logistic_r= LR.fit(X_train, y_train)
        y_sk= logistic_r.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_own)
        precision = precision_score(y_test, y_own)
        recall = recall_score(y_test, y_own)
        own_accuracy.append(accuracy)
        own_precision.append(precision)
        own_recall.append(recall)
        
        accuracy = accuracy_score(y_test, y_sk)
        precision = precision_score(y_test, y_sk)
        recall = recall_score(y_test, y_sk)
        sk_accuracy.append(accuracy)
        sk_precision.append(precision)
        sk_recall.append(recall)

#    print("Own acc:", own_accuracy, "sk_acc:", sk_accuracy)
#    print("Own prec:", own_precision, "sk_prec:", sk_precision)
#    print("Own recall:", own_recall, "sk_recall:", sk_recall)
        
    print("Mean Accuracy")
    print("Own code:", np.mean(np.array(own_accuracy)), "|| SK Learn package:", np.mean(np.array(sk_accuracy)))
    print("Mean Precision")
    print("Own code:", np.mean(np.array(own_precision)), "|| SK Learn package:", np.mean(np.array(sk_precision)))
    print("Mean recall")
    print("Own code:", np.mean(np.array(own_recall)), "|| SK Learn package:", np.mean(np.array(sk_recall)))
    
    
if __name__=="__main__":
    main()