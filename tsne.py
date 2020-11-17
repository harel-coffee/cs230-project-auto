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
from sklearn.manifold import TSNE

# Models
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

plt.style.use('seaborn')


# Constants
PATH_TO_DATA_FOLDER = '../CS 230 Data/Metabolites/'
PATH_TO_TRAIN_DATA = PATH_TO_DATA_FOLDER + 'Metab_psych_data.csv'
# PATH_TO_TEST_DATA = PATH_TO_DATA_FOLDER + 'test_set.csv'
# PATH_TO_DEV_DATA = PATH_TO_DATA_FOLDER + 'dev_set.csv'

DEPRESSION_CUTOFF = 19 # Any BDI score equal or above this will be marked as "depressed"

################################################################################################################

# (1) Read data from CSVs
train_df = pd.read_csv(PATH_TO_TRAIN_DATA)
# test_df = pd.read_csv(PATH_TO_TEST_DATA)
# dev_df = pd.read_csv(PATH_TO_DEV_DATA)

# (2) Create Train/Test/Dev splits
x_train, y_train = train_df.iloc[:,6:], train_df.iloc[:, 4]
# x_test, y_test = test_df.iloc[:,2:], test_df.iloc[:, 1]
# x_dev, y_dev = dev_df.iloc[:,2:], dev_df.iloc[:, 1]
# Binarize Y values
y_train = y_train >= DEPRESSION_CUTOFF
# y_test = y_test[y_test >= DEPRESSION_CUTOFF]
# y_dev = y_dev[y_dev >= DEPRESSION_CUTOFF]

# (3) Merge all datasets
# x_train.append(x_dev, ignore_index = True)
# x_train.append(x_test, ignore_index = True)
# y_train.append(y_dev, ignore_index = True)
# y_train.append(y_test, ignore_index = True)

# (3) Standardize features
feature_scaler = StandardScaler()
x_train = feature_scaler.fit_transform(x_train)

tsne = TSNE(random_state = 0)
tsne_values = tsne.fit_transform(x_train)
plt.title("t-SNE Plot of Patients by their Metabolomics Data")
plt.scatter(tsne_values[y_train, 0], tsne_values[y_train, 1], c="r", label = f'Depressed')
plt.scatter(tsne_values[y_train == False, 0], tsne_values[y_train == False, 1], c="g", label = f'Healthy')
plt.legend()
plt.show()


PATH_TO_DATA_FOLDER = '../data_sets/'
type_of_data = 'metab'
PATH_TO_TRAIN_DATA = PATH_TO_DATA_FOLDER + f'df_train_PCA_dict_df_{type_of_data}.csv'

train_df = pd.read_csv(PATH_TO_TRAIN_DATA)

# (2) Create Train/Test/Dev splits
x_train, y_train = train_df.iloc[:,1:], train_df.iloc[:, 0]

# (3) Standardize features
feature_scaler = StandardScaler()
x_train = feature_scaler.fit_transform(x_train)


tsne = TSNE(random_state = 0)
tsne_values = tsne.fit_transform(x_train)
plt.title("t-SNE Plot of Patients by their Dimensionality-Reduced Metabolomics Data")
plt.scatter(tsne_values[y_train, 0], tsne_values[y_train, 1], c="r", label = f'Depressed')
plt.scatter(tsne_values[y_train == False, 0], tsne_values[y_train == False, 1], c="g", label = f'Healthy')
plt.legend()
plt.show()