from sklearn.model_selection import GridSearchCV
from src.models.naive_bayes import NB
from src.utils.import_data import load_dataset
from src.utils.preprocessing import preprocessing
from src.utils.split_data import split_data
import time

start = time.time()

DATA_PATH = '../../data/FinalStemmedSentimentAnalysisDataset.csv'
DATASET_SIZE = -1
SPLIT_RATIO = 0.2

dataset = load_dataset(DATA_PATH)
print("::-> Dataset loaded")

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

dataset = preprocessing(dataset, delate_nan=True, shuffle=True)

X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=SPLIT_RATIO)

grid_param = {
    'alpha': [0.1, 0.25, 0.75, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 3]}


nb_grid = GridSearchCV(NB(), param_grid=grid_param, n_jobs=-1, cv=5, verbose=2)
nb_grid.fit(X_train, y_train)

print('Train Accuracy : %.3f'%nb_grid.best_estimator_.score(X_train, y_train))
print('Test Accuracy : %.3f'%nb_grid.best_estimator_.score(X_train, y_train))
print('Best Accuracy Through Grid Search : %.3f'%nb_grid.best_score_)
print('Best Parameters : ',nb_grid.best_params_)

end = time.time()

print("\n::-> Test Time : ", end - start)
