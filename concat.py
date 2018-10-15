import pandas as pd
import numpy as np

l = []
for i in range(1,6):
	df = pd.read_csv('results/p'+str(i)+'.csv')
	l.append(df)

r = pd.concat(l)
r.to_csv('results/knn_ga.csv')