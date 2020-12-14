import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train=pd.read_csv('train.csv')       #You should enter the location of you data files 
df_test=pd.read_csv('test.csv')         #in the brackets given.
columns= list(df_train.columns)

def phi(x_train, n):
  X=np.array([[1.0]*(n+1)]*len(x_train))
  for i in range(len(x_train)):
    for j in range(1, n+1):
      X[i][j]= (x_train[i]*X[i][j-1])
  return X


def LassoReg(x_train, y_train, n=1 , l=1, lr=0.05, max_iter=100 ):
  # n= degree of polynomial to be fit. For linear regression it is 1 
  # lr= learning rate default det to 0.05
  # max_iter= maximum number of iterations allowed   
  # l= lambda
  
  l*=1.0
  X= phi(x_train, n)
  Y= y_train.values
  W= np.array([1.0]*(n+1))    #initialising all weights to 1.0
  error=1

  while(error>0.01 and max_iter>0):
    max_iter-=1
    w_old= W

    for i in range(0,n+1):
      temp=np.matmul(X,w_old)-Y
      rho= np.mean(temp*X[:, i])
      # print(rho,W)
      if(rho< -l/2):
        W[i]= W[i]-lr*(rho+ l/2)
      elif(rho> l/2):
        W[i]= W[i]-lr*(rho- l/2)
      else:
        W[i]=W[i]-lr*rho
    error= np.mean((Y-np.matmul(X,W))**2)

  # print(W)
  return W,error

def RidgeReg(x_train, y_train, n=1 , l=1, lr=0.05, max_iter=100 ):
  # n= degree of polynomial to be fit. For linear regression it is 1 
  # lr= learning rate default det to 0.05
  # max_iter= maximum number of iterations allowed   
  # l= lambda
  
  l*=1.0
  X= phi(x_train, n)
  Y= y_train.values
  W= np.array([1.0]*(n+1))    #initialising all weights to 1.0
  error=1

  while(error>0.01 and max_iter>0):
    max_iter-=1
    w_old= W

    for i in range(0,n+1):
      temp=np.matmul(X,w_old)-Y
      rho= np.mean(temp*X[:, i])
      # print(rho,W)
      W[i]= W[i]-lr*(rho+ l*W[i])
    error= np.mean((Y-np.matmul(X,W))**2)

  # print(W)
  return W,error

def main():
    lamda= [0.25,0.5,0.75,1]
    #index 0 is for maximum error and index 1 is for minimum error obtained from above
    n=[1,4]
    for N in n:
      training_error_lasso=[]
      testing_error_lasso=[]
      training_error_ridge=[]
      testing_error_ridge=[]
      if(N==n[0]):
        print("For maximum training error:")
      else:
        print("For minimum training error:")
        
      for l in lamda:
        print("For lamda=",l)
        w_lasso,err=LassoReg(df_train[columns[0]],df_train[columns[1]],n=N,l=l, lr=0.05, max_iter=100000)
        training_error_lasso.append(err)
    
        w_ridge,err=LassoReg(df_train[columns[0]],df_train[columns[1]],n=N,l=l, lr=0.05, max_iter=100000)
        training_error_ridge.append(err)
    
        x_test= df_test[columns[0]].values
        y_test= df_test[columns[1]].values
    
        X=phi(x_test, N)
    
        # print(X.shape)
        y_out_lasso= np.matmul(X, w_lasso)
        testing_error_lasso.append(np.dot((y_out_lasso-y_test).T, (y_out_lasso-y_test)) /len(y_out_lasso))
    
        y_out_ridge= np.matmul(X, w_ridge)
        testing_error_ridge.append(np.dot((y_out_ridge-y_test).T, (y_out_ridge-y_test)) /len(y_out_ridge))
    
        fig,ax= plt.subplots(1,2)
        ax[0].scatter(x_test, y_out_lasso)
        ax[0].set(title="Lasso Regression", xlabel="Feature", ylabel="Value")
        ax[1].scatter(x_test, y_out_ridge)
        ax[1].set(title="Ridge Regression", xlabel="Feature", ylabel="Value")
        plt.show()
    
      fig1, ax1= plt.subplots(1,2)
      ax1[0].plot(lamda, training_error_lasso, label= 'Training Error')
      ax1[0].plot(lamda, testing_error_lasso, label= 'Testing Error')
      ax1[0].set(title="Error Vs lamda for Lasso Regression", xlabel="lamda", ylabel="Error")
      ax1[1].plot(lamda, training_error_ridge, label= 'Training Error')
      ax1[1].plot(lamda, testing_error_ridge, label= 'Testing Error')
      ax1[1].set(title="Error Vs lamda for Lasso Regression", xlabel="lamda", ylabel="Error")
      plt.legend()
      plt.show()
      
if __name__ == "__main__":
    main()