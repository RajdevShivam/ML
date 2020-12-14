import numpy as np
import pandas as pd
import random
import sys

def distance(x, y):     
# =============================================================================
#     This function returns the distance
#     between two vectors as the negative exponential of their cosine similarity
# =============================================================================
    
    # first calculate the cosine similarity
    cos= np.dot(x,y)/((np.dot(x,x)**0.5)*(np.dot(y,y)**0.5))
    #setting distance as negative exponent of cosine similarity
    dis= np.exp(-cos)
    return dis

def initialisePoints(dataSet,k):
  locations= random.sample(range(0,dataSet.shape[0]), k)
  initialPoints= []
  for i in locations:
    initialPoints.append(dataSet[i])
  return initialPoints

def UpdateMean(n,mean,point):
    mean= ((n-1)*mean+point)/n
    return mean

def Classify(means,item): 
  # Classify item to the mean with minimum distance     
  minimum = sys.maxsize
  index = -1
  #print(means.shape)
  for i in range(means.shape[0]): 
    # Find distance from item to mean 
    dis = distance(item, means[i])
    #print(dis)  
    if (dis < minimum): 
      minimum = dis
      index = i
  return index 
    
def CalculateKMeans(k,dataSet,maxIterations=1000):
  # Initialize means at random points 
  means = np.array(initialisePoints(dataSet,k))

  clusterSizes= [0]*means.shape[0] 
  # An array to hold the cluster an item is in 
  belongsTo = [0]*dataSet.shape[0] 

  while(maxIterations):
    maxIterations-=1
    noChange = True # If no change of cluster occurs, halt
 
    for i in range(0,dataSet.shape[0]): 
      item = dataSet[i]
      # Classify item into a cluster and update the corresponding means.         
      index = Classify(means,item)
      #print(index)
      clusterSizes[index] += 1
      #cooccurrence_matrix[item[-1]][index]+=1
      means[index] = UpdateMean(clusterSizes[index],means[index],item)
      # Item changed cluster 
      if(index != belongsTo[i]): 
        noChange = False
      belongsTo[i] = index
      # Nothing changed, return 
      if (noChange): 
            break
    #print(cooccurrence_matrix)
  return means,belongsTo

def findClusters(n,k, belongsTo):
    cluster={}
    for i in range(n):
        try:
            cluster[belongsTo[i]].append(i)
        except:
            cluster[belongsTo[i]]=[i]
    return cluster

def main():
    # Reading the tfidf matrix
    k=8
    df= pd.read_csv("tfidf.csv").drop('Unnamed: 0', axis=1)
    
    #Calling KMean function
    means, belongsTo= CalculateKMeans(k,np.array(df))
    
    #Printing in file
    cluster= findClusters(df.shape[0],k, belongsTo)
    pair=[]
    for i in range(k):
        pair.append((i,cluster[i][0]))
    pair.sort(key= lambda x:x[1]) 
    
    file= open("kmeans.txt", "w+")
    for i in pair:
        file.writelines(str(cluster[i[0]]))
        file.write("\n")
    file.close()       
    
if __name__=="__main__":
    main()