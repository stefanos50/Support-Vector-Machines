import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
import data_preprocessing
import helper_methods
from sklearn.neighbors import KNeighborsClassifier
import time

train_times = []
train_accuracy = []
test_accuracy = []

#train the model and calculate the train time
def train_model(x_train,y_train,k,weights,algorithm,leafsize,p,metric):
    start = time.time()
    model = KNeighborsClassifier(n_neighbors = k,weights=weights,algorithm=algorithm,leaf_size=leafsize,p=p,metric=metric).fit(x_train, y_train)
    stop = time.time()
    return model,(stop-start)

#train the model, calculate the prediction scores for train,test and show the results
def run(x_train,y_train,x_test,y_test):
    m,t = train_model(x_train,y_train,4,"distance","ball_tree",1,1,"minkowski")
    train_accuracy.append(m.score(x_train, y_train))
    test_accuracy.append(m.score(x_test, y_test))
    train_times.append(t)

    print("Train time: "+str(float(train_times[-1])))
    print("Train accuracy: "+str(train_accuracy[-1]))
    print("Test accuracy: "+str(test_accuracy[-1]))

    helper_methods.per_class_accuracy(x_test, y_test, m, max(y_test))
    #helper_methods.plot(x_train,y_train,m,"Train Dataset")
    #helper_methods.plot(x_test,y_test,m,"Test Dataset")

#k cross fold validation technique if test and train datasets are not predefined
def K_Fold_Validation(data,labels,N):
    data_array = np.array(data)
    labels_array = np.array(labels)
    kf = KFold(n_splits=N)
    for train_index, test_index in kf.split(data):
        train_data_x, test_data_x = data_array[train_index], data_array[test_index]
        train_data_y, test_data_y = labels_array[train_index], labels_array[test_index]

        run(train_data_x,train_data_y,test_data_x,test_data_y)
    helper_methods.show_folds_result(train_times,train_accuracy,test_accuracy)

def optimize_parameters(x_data,y_data,param_grid):
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3,cv=5,n_jobs=-1)
    grid.fit(x_data, y_data)
    print(grid.best_params_)
    print(grid.best_estimator_)

#iris dataset
#x,y = data_preprocessing.get_iris_dataset("G:\\iris.data.csv")
#K_Fold_Validation(x,y,10)

#shuttle dataset
#x_train,y_train = data_preprocessing.get_shuttle_dataset("G:\\shuttle_train..txt")
#x_test,y_test = data_preprocessing.get_shuttle_dataset("G:\\shuttle_test.txt")
#run(np.array(x_train),np.array(y_train), np.array(x_test), np.array(y_test))

#eeg emotions detection dataset
x,y = data_preprocessing.get_eeg_biosignals("G:\\emotions.csv")
K_Fold_Validation(x,y,10)


param_grid = {'n_neighbors': [2,3,4,5,6,7,8],
              'algorithm': ['ball_tree','kd_tree','brute','auto'],
              'leaf_size': [1,5,10,30],
              'p':[1,2,3,4],
              'weights':['uniform','distance']}
#optimize_parameters(x,y,param_grid)