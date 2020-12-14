All the codes are written in python IDE Spyder(Python 3.7)
Make sure libraries like pandas, numpy, matplotlib, sklearn etc are already installed.

Task A:Dataset Preparation
1. Code for this is "16AE30018_ML_A3_TA.py" in src folder.
2. Make sure the filename or path which is essentially fed to "pd.read_csv('filename')" is written properly.
3. Put suitable name or path of the csv files consisting of generated dataset tfidf on the lines "to_csv('filename')"

Task B: Agglomerative Clustering
1. Code for this is "16AE30018_ML_A3_TB.py" in src folder.
2. Make sure the filename or path which is essentially fed to "pd.read_csv('filename')" is written properly. Here we have used "tfidf.csv" generated from Task A
3. The output for this is in clusters folder with name ‘agglomerative.txt’

Task C: KMeans Clustering
1. Code for this is "16AE30018_ML_A3_TC.py" in src folder.
2. Make sure the filename or path which is essentially fed to "pd.read_csv('filename')" is written properly. Here we have used "tfidf.csv" generated from Task A
3. The output for this is in clusters folder with name ‘kmeans.txt’

Task D: Attribute Reduction by Principal Component Analysis
1. Code for this is "16AE30018_ML_A3_TD.py" in src folder.
2. Make sure the filename or path which is essentially fed to "pd.read_csv('filename')" is written properly. Here we have used "tfidf.csv" generated from Task A
3. The output for this is in clusters folder with name ‘kmeans_reduced.txt’ and ‘agglomerative_reduced.txt’
4. Please make sure before running this "16AE30018_ML_A3_TB.py", "16AE30018_ML_A3_TC.py", "16AE30018_ML_A3_TD.py" are in same folder or proper paths to them are given
as functions from task B and C are used in this file.




