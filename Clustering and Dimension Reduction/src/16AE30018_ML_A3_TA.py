import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


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

def main():
    #Reading the document
    df= pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
    
    # Dropping the 14th row of the data as it was empty
    df= df.drop([13], axis=0)
    
    #Removing the chapter number from the data
    df['Unnamed: 0']= df['Unnamed: 0'].str.split("_",n=1,expand=True)[0]
    
    #Calculating the tf-idf matrix from DTM
    X= df.drop(['Unnamed: 0'], axis=1)
    
    t= TfidfTransformer(norm='l2', use_idf= True, smooth_idf= True, sublinear_tf= False)
    tfidf= t.fit_transform(X)
    tfidf= tfidf.toarray()
    tfidf= pd.DataFrame(tfidf)
    
    tfidf.to_csv("tfidf.csv")
    
if __name__=="__main__":
    main()
