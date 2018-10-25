import pandas as pd

# read the dataset
dataset = pd.read_csv("../datasets/covertype_norm_train.csv")
# read the dataset of selected attributes
ga = pd.read_csv('../results/ga_selected_attributes.csv')
# class correlations
class_correlations = dataset.corr(method="pearson")['cover_type']

def fitness(data):
	count = data.count(True)
	mean = 0.0
	for i in range(len(data)):
		if data[i]:
			mean += abs(class_correlations[i])
	return [(mean / count), count]

results = pd.DataFrame(columns = ['fitness', 'n_attr']);
for index, row in ga.iterrows():
	results.loc[index] = fitness( ga.iloc[index,-54:].tolist() )

results.fillna(0, inplace=True)
results.sort_values(by=['fitness', 'n_attr'], ascending=True, inplace=True)
results.to_csv('../results/ga_fitness.csv')