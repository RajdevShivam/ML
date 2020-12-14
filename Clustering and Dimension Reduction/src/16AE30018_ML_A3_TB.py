import numpy as np
import pandas as pd
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

def distanceMatrix(X):
# =============================================================================
#     This function returns the matrix of 
#     distances between any combination of two rows of X
# =============================================================================
    matrix=[]
    for i in range(X.shape[0]):
        row= []
        for j in range(X.shape[0]):
            if(i==j):
                row.append(sys.maxsize)
            else:
                row.append(distance(X[i],X[j]))
        matrix.append(row)
    return matrix

def agglomerativeClustering(matrix,linkage="single"):
    clusters = {}
    row_index = -1
    col_index = -1
    
    array= list(range(0,matrix.shape[0]))  
    clusters[0] = array.copy()

    for k in range(1, matrix.shape[0]):
        min_val = sys.maxsize
        
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                if(matrix[i][j]<=min_val):
                    min_val = matrix[i][j]
                    row_index = i
                    col_index = j

        #For single linkage
        if(linkage == "single"):
            for i in range(0,matrix.shape[0]):
                if(i != col_index):
                    temp = min(matrix[col_index][i],matrix[row_index][i])
                    matrix[col_index][i] = temp
                    matrix[i][col_index] = temp

        # print("Distance Matrix After Updating:")
        # print(matrix)

        for i in range (0,matrix.shape[0]):
            matrix[row_index][i] = sys.maxsize
            matrix[i][row_index] = sys.maxsize
       
        minimum = min(row_index,col_index)
        maximum = max(row_index,col_index)
        for n in range(len(array)):
            if(array[n]==maximum):
                array[n] = minimum
        #print(array)
        clusters[k] = array.copy()
        
    return clusters

def FindClusters(cluster, numberOfClusters=8 ):
      clusterKey= len(cluster) - numberOfClusters
      uniqueClusters={}
      key=[]
      #print(len(cluster[clusterKey]))
      for i in range(len(cluster[clusterKey])):
        try:
          uniqueClusters[cluster[clusterKey][i]].append(i)
        except:
          uniqueClusters[cluster[clusterKey][i]]= [i]
          key.append(i)
      return uniqueClusters, key

def main():
    
    # Reading the tfidf matrix
    df= pd.read_csv("tfidf.csv").drop('Unnamed: 0', axis=1)
    
    #Separating distance matrix from labels
    matrix= np.array(distanceMatrix(df.values))
    
    #Function call for clustering and finding n clusters
    cluster= agglomerativeClustering(matrix)
    uniqueClusters, key= FindClusters(cluster, 8)
    
    # Writing the code in a file
    file= open("agglomerative.txt", "w+")
    for i in key:
        file.writelines(str(uniqueClusters[i]))
        file.write("\n")
    file.close()
    
if __name__=="__main__":
    main()