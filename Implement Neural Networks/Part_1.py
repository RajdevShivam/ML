import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def sigmoid(z):
    s= 1/(1+ np.exp(-z))
    return s

def sigmoid_der(z):
    return sigmoid(z) *(1-sigmoid(z))

def ReLu(z):
    return np.maximum(z,0)

def ReLu_der(z):
    z[z>0]=1
    z[z<=0]=0
    return z

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def softmax_der(z):
    return softmax(z)*(1-softmax(z))

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

def data_loader(X, y, batch_size=32): 
    mini_batches = []  
    n_minibatches = int(X.shape[0]/batch_size)
    i = 0
  
    for i in range(n_minibatches + 1): 
        X_mini = X[i * batch_size:(i + 1)*batch_size, :]
        Y_mini = y[i * batch_size:(i + 1)*batch_size, :]
        mini_batches.append((X_mini, Y_mini)) 
        
    if(X.shape[0] % batch_size != 0): 
        X_mini = X[i * batch_size:X.shape[0]] 
        Y_mini = y[i * batch_size:X.shape[0]] 
        mini_batches.append((X_mini, Y_mini)) 
        
    return mini_batches 

def weight_initialiser(neuron_no, input_shape):
    W= np.random.uniform(low=-1.0, high=1.0, size=(neuron_no, input_shape))
    return W

def forward(X, W, func):
    out=[]
    h=func(np.dot(X, W[0].T))
    out.append(h)
    
    for i in range(1,len(W)-1):
        h=func(np.dot(out[i-1], W[i].T))
        out.append(h)
        
    h= softmax(np.dot(out[-1], W[-1].T))
    out.append(h)
    return out

def backpropagation(X, y, out, W, func, func_d, lr=0.01):
# application of the chain rule to find derivative of the loss function
    dw= (out[-1]-y)
    temp= np.dot(dw , W[-1])
    W[-1]= W[-1]-lr*(np.dot(out[-2].T,dw)).T
    i= len(W)-1
    while(i>1):
        i-=1
        temp= temp*func_d(out[i])
        temp1= np.dot(temp, W[i])
        W[i]= W[i]-lr*(np.dot(out[i-1].T, temp)).T
        temp=temp1
    W[0]= W[0]- lr*(np.dot(X.T, func_d(out[0])*temp)).T
    
def predict(X, W, func):
    out= forward(X, W, func)
    return out[-1]

def NN1(max_iters= 200):
    
    X_train, X_test, y_train, y_test= preprocess()
    
    wh1= weight_initialiser(64, X_train.shape[1])
    wo= weight_initialiser(3, wh1.shape[0])
    W=[wh1, wo]
    
    training_err=[]
    test_err=[]
    
    mini_batches = data_loader(X_train, y_train)
    for itr in range(max_iters):
        for X_mini, y_mini in mini_batches:
            out= forward(X_mini, W, sigmoid)
            backpropagation(X_mini, y_mini, out, W,sigmoid,sigmoid_der , lr=0.01)
            
        if( itr%10 == 0):
            h=predict(X_train,W,sigmoid)
            loss = np.mean(-y_train * np.log(h))
            training_err.append(loss)
            
            h=predict(X_test,W,sigmoid)
            loss = np.mean(-y_test * np.log(h))
            test_err.append(loss)
    
    print("For Part 1A")
    print("Training score:", training_err[-1])
    print("Testing score:", test_err[-1])
    
    num=list(range(0,max_iters,10))
    plt.plot(num , training_err, color='r') 
    plt.plot(num, test_err, color='b') 
    plt.xlabel("Number of iterations") 
    plt.ylabel("Cost") 
    plt.show() 
    
def NN2(max_iters= 200):
    
    X_train, X_test, y_train, y_test= preprocess()
    
    wh1= weight_initialiser(64, X_train.shape[1])
    wh2= weight_initialiser(32, wh1.shape[0])
    wo= weight_initialiser(3, wh2.shape[0])
    W=[wh1,wh2, wo]
    
    training_err=[]
    test_err=[]
    
    mini_batches = data_loader(X_train, y_train)
    for itr in range(max_iters):
        for X_mini, y_mini in mini_batches:
            out= forward(X_mini, W, ReLu)
            backpropagation(X_mini, y_mini, out, W,ReLu,ReLu_der , lr=0.01)
            
        if( itr%10 == 0):
            h=predict(X_train,W,ReLu)
            loss = np.mean(-y_train * np.log(h))
            training_err.append(loss)
            
            h=predict(X_test,W,ReLu)
            loss = np.mean(-y_test * np.log(h))
            test_err.append(loss)
    
    print("For Part 1B")
    print("Training score:", training_err[-1])
    print("Testing score:", test_err[-1])
    
    num=list(range(0,max_iters,10))
    plt.plot(num , training_err, color='r') 
    plt.plot(num, test_err, color='b') 
    plt.xlabel("Number of iterations") 
    plt.ylabel("Cost") 
    plt.show() 
    
if __name__=="__main__":
    NN1()
    NN2()
            