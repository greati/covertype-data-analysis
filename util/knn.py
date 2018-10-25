# Import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Original dataset
dataset_train = pd.read_csv('../datasets/covertype_norm_train.csv')
dataset_test = pd.read_csv('../datasets/covertype_norm_test.csv')

# Targets
target_train = dataset_train.iloc[:,-1]
# Dataset without classes
data_train = dataset_train.iloc[:,:-1]

# Targets
target_test = dataset_test.iloc[:,-1]
# Dataset without classes
data_test = dataset_test.iloc[:,:-1]

# Read Selected Attributes of GA
ga  = pd.read_csv('../results/ga_selected_attributes.csv')

# Defining K's
ks = [1,3,5,7,9]
acc_list = ['k1','k3','k5','k7','k9']

def perform_knn(attr, start, end):
    '''
    Performs knn for a given dataset.
    '''
    columns = attr.columns.tolist()[:-54] + acc_list
    new_df = pd.DataFrame(columns=columns)
    
    for index in range(start, end):
        vector = attr.iloc[index,-54:].tolist()
        sliced_train = data_train.iloc[:, vector]       
        sliced_test = data_test.iloc[:, vector]
        
        # Perform knn
        r = []
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn.fit(sliced_train, target_train)
            r.append(knn.score(sliced_test, target_test))
        
        new_df.loc[index] = attr.iloc[0,:-54].tolist() + r
        print(index)
    
    return new_df

r =  [(0, 486), (486, 972), (972, 1458), (1458, 1944), (1944, 2430)]

p = int(input())
start, end = r[p-1]

df = perform_knn(ga, start, end)
df.to_csv('../results/p' + str(p) + '.csv')