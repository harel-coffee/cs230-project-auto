import pandas as pd
from time import time
from joblib import dump, load
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# Data processing / Metrics
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Models
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

plt.style.use('seaborn')

# Constants
TYPES_OF_DATA = ['card', 'cyto', 'lipid', 'metab', 'metpan', ]


type_of_data = 'metab'

PATH_TO_DATA_FOLDER = '../data_sets/'
PATH_TO_TRAIN_DATA = PATH_TO_DATA_FOLDER + f'df_train_PCA_dict_df_{type_of_data}.csv'
PATH_TO_TEST_DATA = PATH_TO_DATA_FOLDER + f'df_test_PCA_dict_df_{type_of_data}.csv'
PATH_TO_DEV_DATA = PATH_TO_DATA_FOLDER + f'df_dev_PCA_dict_df_{type_of_data}.csv'

################################################################################################################

# (1) Read data from CSVs
train_df = pd.read_csv(PATH_TO_TRAIN_DATA)
test_df = pd.read_csv(PATH_TO_TEST_DATA)
dev_df = pd.read_csv(PATH_TO_DEV_DATA)

# (2) Create Train/Test/Dev splits
x_train, y_train = train_df.iloc[:,1:12], train_df.iloc[:, 0]
x_test, y_test = test_df.iloc[:,1:12], test_df.iloc[:, 0]
x_dev, y_dev = dev_df.iloc[:,1:12], dev_df.iloc[:, 0]
# Merge dev + train sets since SKLearn does CV for us
x_train.append(x_dev, ignore_index = True)
y_train.append(y_dev, ignore_index = True)


# (3) Standardize features
feature_scaler = StandardScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

print("Dataset Shapes:", train_df.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape, "TRUE Y: ", np.sum(y_train))

mps = [
	{
		'model' : RandomForestClassifier,
		'params' : {
			'max_depth': [10, 20, 50, 60, 90, 100, None],
			'max_features': ['auto', 'sqrt'],
			'min_samples_leaf': [1, 2, 4],
			'min_samples_split': [2, 5, 10],
			'n_estimators': [50, 100, 300, 500],
			'class_weight' : [ None, 'balanced', ]
		}
	},
	{
		'model' : SVC,
		'params' : {
		    'kernel': ['linear', 'poly', 'rbf', 'sigmoid' ],
		    'C' : [0.1, 0.5, 1],
		}
	},
	{
		'model' : KNeighborsClassifier,
		'params' : {
		    'n_neighbors': [1, 2, 3, 4],
		}
	},
	{
		'model' : GradientBoostingClassifier,
		'params' : {
		    'n_estimators': [50, 100, 300, 500]
		}
	},
	{
		'model' : AdaBoostClassifier,
		'params' : {
		    'n_estimators': [50, 100, 300, 500]
		}
	},
	{
		'model' : MLPClassifier,
		'params' : {
		    'hidden_layer_sizes': [(50), (50,50), (50,50,50),],
		    'alpha' : [ 0.0001, 0.001, 0.01 ],
		    'max_iter' : [500],
		}
	},
]
N_FOLDS = 5
SCORING_METRIC = 'f1'
best_score, best_model = None, None
results = []
for m in mps:
	model = m['model']
	params = m['params']
	print("---- ", model, " ----")
	# Grid Search CV
	sss = StratifiedShuffleSplit(n_splits = N_FOLDS, test_size = 0.2, random_state=0)
	clf = GridSearchCV(model(), params, 
						scoring = SCORING_METRIC, 
						cv = sss,
						n_jobs = -1)
	clf.fit(x_train, y_train)
	# Get best model
	val_model = model(**clf.best_params_)
	val_model.fit(x_train, y_train)
	# Make predictions on x_test
	y_preds = val_model.predict(x_test)
	accuracy = accuracy_score(y_preds, y_test)
	f1 = f1_score(y_preds, y_test)
	# Save results
	results.append(clf.cv_results_)
	print("Validated Model | Accuracy: ", accuracy, "F1: ", f1)
	score = f1
	if best_score is None or score < best_score:
		best_score = score
		best_model = val_model

# Print best model
print("----")
print("Best Model:", best_model)
print("F1 Score:", best_score)
print("Params:", best_model.get_params())

# Save best model
dump(best_model, 'best_model.joblib')

# Plot results
plot_data = [ r['mean_test_score'] for r in results ]
fig, ax = plt.subplots()
ax.set_title('Cross-Validation Test Scores for ' + type_of_data)
ax.set_xlabel('Model')
ax.set_ylabel('F1 Score')
ax.boxplot(plot_data, vert = True, labels = [ type(x['model']()).__name__ for x in mps ])
plt.show()
