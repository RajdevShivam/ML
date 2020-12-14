#Loading pandas as pd

import pandas as pd

def minMaxScaling(data):
    # Function used for doing min-max scaling
  dataf=((data-data.min())/(data.max()-data.min()))
  return dataf

def ZScoreNormalization(data):
  dataf= ((data-data.mean())/data.std())
  return dataf

def main():
    # Reading CSV Files in the project
    # Keep the code and file in the same folder  before running it
    
    dataA= pd.read_csv("data/winequality-red.csv", sep=";")
    dataB= pd.read_csv("data/winequality-red.csv", sep=";")
    
    
    # Creation of dataset A
    dataA.loc[dataA['quality']<=6, 'quality'] = 0
    dataA.loc[dataA['quality']>6, 'quality'] = 1
    
    dataA= minMaxScaling(dataA)
    dataA.to_csv("dataA.csv", index=False) #Saving dataset A in csv format for future use
    
    #Creation of dataset B
    dataB.loc[dataB['quality']<5, 'quality'] = 0
    dataB.loc[dataB['quality']==5, 'quality'] = 1
    dataB.loc[dataB['quality']==6, 'quality'] = 1
    dataB.loc[dataB['quality']>6, 'quality'] = 2
    
    #Drop quality as we don't need to change it and thus we can add it later

    data= dataB.drop(['quality'], axis=1)
    quality= dataB['quality']
    dataB= ZScoreNormalization(data)
    
    r= (dataB.max(axis=0)-dataB.min(axis=0))/4
    t1= dataB.min(axis=0)+r
    t2= t1+r
    t3= t2+r
    
    for col in dataB.columns:
        dataB.loc[(dataB[col]>t3[col]), col]=3
        dataB.loc[((dataB[col]<=t3[col]) & (dataB[col]>t2[col])), col] = 2
        dataB.loc[((dataB[col]<=t2[col]) & (dataB[col]>t1[col])), col] = 1
        dataB.loc[dataB[col]<=t1[col], col] = 0
    
    dataB['quality']= quality # Adding back quality column to the dataframe
    dataB.to_csv("dataB.csv", index=False) #Saving dataset B in csv format for future use
    
    
if __name__=="__main__":
    main()