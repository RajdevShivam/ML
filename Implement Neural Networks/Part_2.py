import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

def preprocess():
    test= pd.read_csv("test_data.csv")
    train= pd.read_csv("train_data.csv")
    
    y_test=test[test.columns[-3:]].values
    X_test=test.drop(test.columns[-3:], axis=1).values
    X_test= np.insert(X_test, X_test.shape[1], 1, axis=1)
    
    y_train=train[train.columns[-3:]].values
    X_train=train.drop(train.columns[-3:], axis=1).values
    X_train= np.insert(X_train, X_train.shape[1], 1, axis=1)
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test= preprocess()
    ml_p1A= MLPClassifier(hidden_layer_sizes=(32,), activation='logistic', max_iter=200, solver='sgd', verbose=20, batch_size=32, learning_rate_init=0.01)
    ml_p1A.fit(X_train, y_train)
    
    print("\n For part A:")
    print("Training Score: %f" %ml_p1A.score(X_train, y_train))
    print("Testing Score: %f" %ml_p1A.score(X_test, y_test))
    
    ml_p1B= MLPClassifier(hidden_layer_sizes=(64,32,), activation='relu', max_iter=200, solver='sgd', verbose=20, batch_size=32, learning_rate_init=0.01)
    ml_p1B.fit(X_test, y_test)
    
    print("\n For part B:")
    print("Training Score: %f" %ml_p1A.score(X_train, y_train))
    print("Testing Score: %f" %ml_p1A.score(X_test, y_test))
    
    
if __name__=="__main__":
    main()