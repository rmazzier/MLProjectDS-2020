##################################################
# Process the data here, if needed
##################################################
#For better numerical stability is advisible to divide by 255
x_train=x_train/255
x_valid=x_valid/255


##################################################
# Implement you model here
##################################################

import joblib

###### KNN #####
#scaling the data (StandardScaler & MinMaxScaler tested) is maybe not really necessary
#since all features have the same unit of measurement (int in [0;255]).
#Trying with and without standardization, the train and valid set perfomance is
#the same, so we do not apply any standardization.
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()#MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_train=x_train_scaled
x_valid=x_valid_scaled


#***************************************************************************
#to run only if you want to work with less data to speed up computation in experimental phase
x_train=x_train[0:1000]
y_train=y_train[0:1000]
x_valid=x_valid[0:1000]
y_valid=y_valid[0:1000]
#*******************************************************************************

#use grid search to find the best value of k (='n_neighbors'). 
#We do not look for the best value of 'p' in the L_p metrics, because theorical 
#results have prooved that the meaningfulness of the L_p norm worsens faster within 
#increasing dimensionality for higher values of p. Thus, for a given problem with a 
#fixed (high) value for the dimensionality d, it is preferable to use lower values of p.
#This means that the L1 distance metric (Manhattan distance metric) is the most preferable 
#for high dimensional applications, followed by the Euclidean metric (L2).  
#For reference see:
#Charu C. Aggarwal, Alexander Hinneburg, Daniel A. Keim, "On the Surprising Behavior of Distance Metrics in High Dimensional Space"
#Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, Uri Shaft, "When Is 'Nearest Neighbor' Meaningful?"
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors': list(range(1,21))}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
grid_search = GridSearchCV(KNeighborsClassifier(p=1), param_grid, cv=cv,n_jobs=-1)
grid_search.fit(x_train, y_train)

# save the model to disk
filename = 'grid_search.sav'
joblib.dump(grid_search, filename)

# to load the model from disk
filename = 'grid_search.sav'
grid_search = joblib.load(filename)

# convert the grid_search model in a dataframe to see easely the results
results_grid_search_knn = pd.DataFrame(grid_search.cv_results_)
results_grid_search_knn=results_grid_search_knn[['params','mean_test_score','rank_test_score']]
# get the optimal hyperparameter 'n_neighboors'
k_star=grid_search.best_params_['n_neighbors']

#summary of model and performaces
print("Best estimator:\n{}".format(grid_search.best_estimator_))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
print("Validation set score: {:.3f}".format(grid_search.score(x_valid, y_valid)))



#To recognize an item, we don't look at the single pixel, but at the overall image.
#So extracting Principal Components (PCs) and using them to identify different items 
#is an attempt to mimic how humans identify items.
#Using the optimal value of k found before, we look how much varies the accuracy
#in train and valid set increasing the n° of PCs used to approximate each sample.
#Around 20 PCs, the Accuracy curve starts being flat, meaning that adding further 
#PCs increases just a little the performance, so it's not worth to add them.
from sklearn.decomposition import PCA
test_accuracy = []
PC_number = range(1, 101)
for i in PC_number:
    pca = PCA(n_components=i, whiten=True, random_state=0).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_valid_pca = pca.transform(x_valid)
    knn = KNeighborsClassifier(n_neighbors=k_star,p=1)
    knn.fit(x_train_pca, y_train)
    # record generalization accuracy
    test_accuracy.append(knn.score(x_valid_pca, y_valid.reshape(-1,1)))

plt.plot(PC_number, test_accuracy, label="valid accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n° PC")
plt.legend()  

#save the figure
plt.savefig('Accuracy VS number_PCs.png')

#load the figure
import matplotlib.image as mpimg
img = mpimg.imread('Accuracy VS number_PCs.png')
#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)
plt.axis('off')
plt.imshow(img,interpolation='spline36')

#optimal number of PC
PCs_star=np.argmax(test_accuracy)+1
#observing the plot, the accuracy growth becomes flat approx. after the 
#first 20 PCs. So to have an easier model, with less features, we 
#decide to use 20 PCs.
PCs_star=20
valid_set_accuracy=test_accuracy[PCs_star-1]

#compute the first 20 PCs for x_train and x_valid
pca = PCA(n_components=20, whiten=True, random_state=0).fit(x_train)
x_train_pca = pca.transform(x_train)
x_valid_pca = pca.transform(x_valid)

# save the model to disk
filename = 'pca.sav'
joblib.dump(pca, filename)

# to load the model from disk
filename = 'pca.sav'
pca = joblib.load(filename)

#train a knn model with k=k_star and n° PCs=20
knn = KNeighborsClassifier(n_neighbors=k_star,p=1)
knn.fit(x_train_pca, y_train)
valid_set_accuracy_20=knn.score(x_valid_pca, y_valid.reshape(-1,1))
#this is the same model with 20 PCs trained and tested in above for cycle,
#so test_accuracy[19] == valid_set_accuracy_20

# save the model to disk
filename = '5nn.sav'
joblib.dump(knn, filename)
# to load the model from disk
knn = joblib.load(filename)

#plot the PCs
print("pca.components_.shape: {}".format(pca.components_.shape))
fix, axes = plt.subplots(4, 5, figsize=(15, 12),subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape((28,28)),cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))
plt.savefig('first 20 PCs.png')

#load the figure
img = mpimg.imread('first 20 PCs.png')
plt.axis('off')
plt.imshow(img,interpolation='spline36')

##################################################
# Evaluate the model here
##################################################

# Use this function to evaluate your model
y_pred=knn.predict(x_valid_pca)
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()

# Report the accuracy in the train and validation sets.
valid_set_accuracy_20_helper=accuracy(y_pred,y_valid)




####### Decision Tree & Random Forest using PCs #####

#Using PCA, we reduced the features' dimension from 784 to 20. Now, we want to apply
#a 'Decision Tree' to classify the items using only the 20 PCs extracted before.
from sklearn.tree import DecisionTreeClassifier
#choosing the parameter 'max_depth' high enough, it's possible to classify 
#perfectly each class in the training set, reaching an accuracy of 100%. But
#then the performance in the test set is going to be much smaller, due to overfitting.
tree = DecisionTreeClassifier(max_depth=15,random_state=0)
tree.fit(x_train_pca, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(x_train_pca, y_train)))
print("Accuracy on validation set: {:.3f}".format(tree.score(x_valid_pca, y_valid)))

#A random forest is essentially a collection of decision trees, where each tree is slightly different from
#the others. The idea behind random forests is that each tree might do a relatively
#good job of predicting, but will likely overfit on part of the data. If we build many
#trees, all of which work well and overfit in different ways, we can reduce the amount
#of overfitting by averaging their results. 
from sklearn.ensemble import RandomForestClassifier
#'n_estimators':number of trees to build(larger is always better. Averaging more trees will
#                                        yield a more robust ensemble by reducing overfitting.)
#'max_features': number of randomly selected features among which the algo looks for the best one
#                it’s a good rule of thumb to use the default values: max_features=sqrt(n_features) 
forest = RandomForestClassifier(n_estimators=100,
                                max_depth=15,
                                max_features=5, 
                                random_state=0,
                                n_jobs=-1)
forest.fit(x_train_pca, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(x_train_pca, y_train)))
print("Accuracy on valid set: {:.3f}".format(forest.score(x_valid_pca, y_valid)))

#use grid search to find the best value of 'max_depth' and 'max_features'
param_grid = {'max_depth': list(range(1,21)),
              'max_features': list(range(1,21))}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100,
                                                  random_state=0,
                                                  n_jobs=-1),
                           param_grid, cv=cv,n_jobs=-1)
grid_search.fit(x_train_pca, y_train)

# save the model to disk
filename = 'grid_search_forest.sav'
joblib.dump(grid_search, filename)

# to load the model from disk
filename = 'grid_search_forest.sav'
grid_search = joblib.load(filename)

# convert the grid_search model in a dataframe to see easely the results
results_grid_search_randForest = pd.DataFrame(grid_search.cv_results_)
results_grid_search_randForest=results[['params','mean_test_score','rank_test_score']]

#summary of model and performaces
print("Best estimator:\n{}".format(grid_search.best_estimator_))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Validation set score: {:.2f}".format(grid_search.score(x_valid_pca, y_valid)))

#train the model with optimal hyperparameters and test it
forest = RandomForestClassifier(n_estimators=100,
                                max_depth=20,
                                max_features=6, 
                                random_state=0,
                                n_jobs=-1)
forest.fit(x_train_pca, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(x_train_pca, y_train)))
print("Accuracy on valid set: {:.3f}".format(forest.score(x_valid_pca, y_valid)))

#problem: the optimal solution is a corner one: max_depth=20
#to be sure that does not exist a better model with higher depth, we expand
#the grid search to values of max_depth higher than 20

#use extension of grid search to find the best value of 'max_depth' and 'max_features'
param_grid = {'max_depth': list(range(20,30)),
              'max_features': list(range(5,11))}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100,
                                                  random_state=0,
                                                  n_jobs=-1),
                           param_grid, cv=cv,n_jobs=-1)
grid_search.fit(x_train_pca, y_train)

# save the model to disk
filename = 'grid_search_restricted_forest.sav'
joblib.dump(grid_search, filename)

# to load the model from disk
filename = 'grid_search_restricted_forest.sav'
grid_search = joblib.load(filename)

# convert the grid_search model in a dataframe to see easely the results
results_grid_search_randForest = pd.DataFrame(grid_search.cv_results_)
results_grid_search_randForest=results_grid_search_randForest[['params','mean_test_score','rank_test_score']]


#summary of model and performaces
print("Best estimator:\n{}".format(grid_search.best_estimator_))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Validation set score: {:.2f}".format(grid_search.score(x_valid_pca, y_valid)))

#train the model with optimal hyperparameters and test it
forest = RandomForestClassifier(n_estimators=100,
                                max_depth=24,
                                max_features=7, 
                                random_state=0,
                                n_jobs=-1)
forest.fit(x_train_pca, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(x_train_pca, y_train)))
print("Accuracy on valid set: {:.3f}".format(forest.score(x_valid_pca, y_valid)))

#Anyway the major advantage of Tree is interpretability, so using a Decision Tree when 
#features have not a clear interpretation (because are PCs) is not meaningfull.
#Better to use KNN with PCs.

##To Be Done (Maybe!!!)
#What we would like is to build hand crafted features to use Tree.