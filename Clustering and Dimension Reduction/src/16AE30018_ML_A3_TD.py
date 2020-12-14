import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
TB = __import__('16AE30018_ML_A3_TB')
TC = __import__('16AE30018_ML_A3_TC')

def main():
    #Reading the data
    df= pd.read_csv("tfidf.csv").drop('Unnamed: 0', axis=1)
    #dimentionality reduction of data
    pca = PCA(n_components=100)
    X= pca.fit_transform(df.values)
    
    #Agglomerative Clustering with reduced data
    matrix= np.array(TB.distanceMatrix(X))
    cluster= TB.agglomerativeClustering(matrix)
    uniqueClusters, key= TB.FindClusters(cluster, 8)
        
        # Writing the code in a file
    file= open("agglomerative_reduced.txt", "w+")
    for i in key:
        file.writelines(str(uniqueClusters[i]))
        file.write("\n")
    file.close()
    
    #KMeans clustering with reduced data
    k=8
    means, belongsTo= TC.CalculateKMeans(k,X)
    cluster= TC.findClusters(df.shape[0],k, belongsTo)
    pair=[]
    for i in range(k):
        pair.append((i,cluster[i][0]))
    pair.sort(key= lambda x:x[1]) 
    
    file= open("kmeans_reduced.txt", "w+")
    for i in pair:
        file.writelines(str(cluster[i[0]]))
        file.write("\n")
    file.close()
    
if __name__=="__main__":
    main()
    