import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df= pd.read_csv("seeds.txt", sep="\t", header= None, names= ["area", "perimeter", "compactness", "length of kernal", "width of kernal", "asymmetry coefficient", "length of kernal groove", "classes"])

for i in range(df.shape[1]-1):
    df.iloc[:,i]=(df.iloc[:,i]-np.mean(df.iloc[:,i]))/np.std(df.iloc[:,i])

y=df[df.columns[-1]]
X= df.drop(df.columns[-1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

test= pd.concat([pd.DataFrame(X_test),pd.DataFrame(y_test)], sort=False, axis=1)
test[["col_0", "col_1", "col_2"]]= pd.get_dummies(test[test.columns[-1]])
test= test.drop(["classes"], axis=1)

train= pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)], sort=False, axis=1)
train[["col_0", "col_1", "col_2"]]= pd.get_dummies(train[train.columns[-1]])
train= train.drop(["classes"], axis=1)

train.to_csv("train_data.csv", index=False)
test.to_csv("test_data.csv", index=False)