# Import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Original dataset
dataset = pd.read_csv('datasets/new_dataset_covertype.csv')

# Targets
target = dataset.iloc[:,-1]
# Dataset without classes
data   = dataset.iloc[:,:-1]

# Read Selected Attributes of GA
ga  = pd.read_csv('results/genetic_algorithm.csv')

# Defining K's
ks = [7,10,15,20,25]
acc_list = ['k7','k10','k15','k20','k25']

def perform_knn(data, attr, start, end):
    columns = attr.columns.tolist()[:-54] + acc_list
    new_df = pd.DataFrame(columns=columns)
    
    for index in range(start, end):
        vector = attr.iloc[index,-54:].tolist()
        sliced_data = data.iloc[:, vector]
        
        # Perform knn
        r = []
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', n_jobs=-1)
            score = cross_val_score(knn, sliced_data, target, cv=10)
            r.append(score.mean())
        
        new_df.loc[index] = attr.iloc[0,:-54].tolist() + r
        print(index)
    
    return new_df

r =  [(0, 486), (486, 972), (972, 1458), (1458, 1944), (1944, 2430)]

p = int(input())
start, end = r[p-1]

df = perform_knn(data, ga, start, end)
df.to_csv('results/p' + str(p) + '.csv')