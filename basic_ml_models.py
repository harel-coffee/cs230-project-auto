import pandas as pd
from time import time
from joblib import dump, load

# Plotting
import matplotlib.pyplot as plt

# Data processing / Metrics
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# Models
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Constants
PATH_TO_DATA_FOLDER = '../data_sets/'
PATH_TO_TRAIN_DATA = PATH_TO_DATA_FOLDER + 'train_set.csv'
PATH_TO_TEST_DATA = PATH_TO_DATA_FOLDER + 'test_set.csv'
PATH_TO_DEV_DATA = PATH_TO_DATA_FOLDER + 'dev_set.csv'

################################################################################################################

# (1) Read data from CSVs
train_df = pd.read_csv(PATH_TO_TRAIN_DATA)
test_df = pd.read_csv(PATH_TO_TEST_DATA)
dev_df = pd.read_csv(PATH_TO_DEV_DATA)

# (2) Create Train/Test/Dev splits
x_train, y_train = train_df.iloc[:,2:], train_df.iloc[:, 1]
x_test, y_test = test_df.iloc[:,2:], test_df.iloc[:, 1]
x_dev, y_dev = dev_df.iloc[:,2:], dev_df.iloc[:, 1]
x_train.append(x_dev, ignore_index = True)
y_train.append(y_dev, ignore_index = True)

# (3) Standardize features
feature_scaler = StandardScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

print("Dataset Shapes:", train_df.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape)

mps = [
	{
		'model' : RandomForestRegressor,
		'params' : {
		    'n_estimators': [50, 100, 300, 500]
		}
	},
	{
		'model' : SVR,
		'params' : {
		    'kernel': ['linear', 'poly', 'rbf', 'sigmoid' ],
		    'C' : [0.1, 0.5, 1],
		}
	},
	{
		'model' : KNeighborsRegressor,
		'params' : {
		    'n_neighbors': [1, 2, 3, 4],
		}
	},
	{
		'model' : GradientBoostingRegressor,
		'params' : {
		    'n_estimators': [50, 100, 300, 500]
		}
	},
	{
		'model' : AdaBoostRegressor,
		'params' : {
		    'n_estimators': [50, 100, 300, 500]
		}
	},
	{
		'model' : MLPRegressor,
		'params' : {
		    'hidden_layer_sizes': [(100), (100,100), (100,100,100),],
		    'alpha' : [ 0.0001, 0.001, 0.01 ],
		}
	},
]
N_FOLDS = 5
SCORING_METRIC = 'neg_mean_squared_error'
best_mse, best_model = None, None
results = []
for m in mps:
	model = m['model']
	params = m['params']
	print("---- ", model, " ----")
	# Grid Search CV
	clf = GridSearchCV(model(), params, scoring = SCORING_METRIC, cv = N_FOLDS, n_jobs = -1)
	clf.fit(x_train, y_train)
	# Get best model
	val_model = model(**clf.best_params_)
	val_model.fit(x_train, y_train)
	# Make predictions on x_test
	y_preds = val_model.predict(x_test)
	mse = mean_squared_error(y_preds, y_test)
	# Save results
	results.append(clf.cv_results_)
	print("Validated Model MSE: ", mse)
	if best_mse is None or mse < best_mse:
		best_mse = mse
		best_model = val_model

# Print best model
print("----")
print("Best Model:", best_model)
print("MSE:", best_mse)
print("Params:", best_model.get_params())

# Save best model
dump(best_model, 'best_model.joblib')

# Plot results
plot_data = [-1 * r['mean_test_score'] for r in results ]
fig, ax = plt.subplots()
ax.set_title('Cross-Validation Test Scores')
ax.set_xlabel('Model')
ax.set_ylabel('MSE')
ax.boxplot(plot_data, vert = True, labels = [ type(x['model']()).__name__ for x in mps ])
plt.show()
ax.boxplot()