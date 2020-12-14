import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train=pd.read_csv('train.csv')       #You should enter the location of you data files 
df_test=pd.read_csv('test.csv')         #in the brackets given.
# print("Train is:")
# print(df_train.head())
# print("Test is:")
# print(df_test.head())
columns= list(df_train.columns)
#print(columns)

plt.scatter(df_train[columns[0]],df_train[columns[1]])
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.title("Training Data")
plt.show()

plt.scatter(df_test[columns[0]],df_test[columns[1]])
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.title("Test Data")
plt.show()

def phi(x_train, n):
  X=np.array([[1.0]*(n+1)]*len(x_train))
  for i in range(len(x_train)):
    for j in range(1, n+1):
      X[i][j]= (x_train[i]*X[i][j-1])
  return X

def polyfit(x_train, y_train, n, lr=0.05, max_iter=100000):

  # n= degree of polynomial to be fit. For linear regression it is 1 
  # lr= learning rate default det to 0.05 
  # max_iter= maximum number of iterations allowed       

  X= phi(x_train.values, n)
  Y= y_train.values

  # print("x_train: ", X[:5])
  # print("y_train: ", Y[:5])

  W= np.array([1.0]*(n+1))    #initialising all weights to 1.0
  error=1

  while(error>0.01 and max_iter>0):
    max_iter-=1
    w_old= W
    for i in range(0,n+1):
      temp=np.matmul(X,w_old)-Y
      W[i]= w_old[i]-lr*np.mean(temp*X[:, i])
    error= np.mean((Y-np.matmul(X,W))**2)

  return W, error

def main():
    training_error=[]
    testing_error=[]
    
    for N in range(1,10):
    
      w, error= polyfit(df_train[columns[0]],df_train[columns[1]],n=N, lr=0.05, max_iter=100000)
      training_error.append(error)
    
      x_test= df_test[columns[0]].values
      y_test= df_test[columns[1]].values
    
      X=phi(x_test, N)
    
      # print(X.shape)
      y_out= np.matmul(X, w)
    
      testing_error.append(np.dot((y_out-y_test).T, (y_out-y_test)) /len(y_out) )
      # print("training_error:", training_error[-1], "testing_error:", testing_error[-1], "Degree:", N)
    
      plt.scatter(x_test, y_out)
      plt.title("Degree of polynomial is: "+ str(N))
      plt.xlabel("Feature")
      plt.ylabel("Label")
      plt.show()
    
    plt.plot(range(1,10), training_error, label= 'Training Error')
    plt.plot(range(1,10), testing_error, label= 'Testing Error')
    plt.title("Error Vs Degree of Polynomial")
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
