
##################################################
# Process the data here, if needed
##################################################

#since we are going to apply regularization, it's advisible to standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_train=x_train_scaled
x_valid=x_valid_scaled


##################################################
# Implement you model here
##################################################

############## Logistic Regression ########
import joblib
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

#define the hyperparameters values for the grid search
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]#inverse of regularization parameter = 1/lambda
solver = ['liblinear', 'saga']#the only two solver which support both l1 and l2 regularization
param_grid = dict(penalty=penalty,
                  C=C,
                  solver=solver)

grid_search = GridSearchCV(estimator=logreg,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=-1)

#to fit the data in a reasonable amount of time, we take 20% of train set to train the model
x_train_reduced=x_train[0:10000]
y_train_reduced=y_train[0:10000]

#fit the grid search
grid_search.fit(x_train_reduced, y_train_reduced)

# save the model to disk
filename = 'grid_search_Logistic.sav'
joblib.dump(grid_search, filename)

# to load the model from disk
filename = 'grid_search_Logistic.sav'
grid_search = joblib.load(filename)

# convert the grid_search model in a dataframe to see easely the results
results_grid_search_logistic = pd.DataFrame(grid_search.cv_results_)
results_grid_search_logistic=results_grid_search_logistic[['params','mean_test_score','rank_test_score']]

#summary of model and performaces
print("Best estimator:\n{}".format(grid_search.best_estimator_))
print("Best parameters: {}".format(grid_search.best_params_))


#train the model with optimal hyperparameters found in the reduced training set, on the full train set
logreg = LogisticRegression(penalty='l2',#ridge regularization
                            C=1,
                            solver='saga',
                            max_iter=100)
logreg.fit(x_train,y_train)


##################################################
# Evaluate the model here
##################################################

# Use this function to evaluate your model
y_pred=logreg.predict(x_valid)
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()

# Report the accuracy in the train and validation sets.
valid_set_accuracy=accuracy(y_pred,y_valid)