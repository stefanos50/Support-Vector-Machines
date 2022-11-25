import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import data_preprocessing
import time
import helper_methods

kernel_names = ['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel']
train_times = [[], [], [], []]
train_accuracy = [[], [], [], []]
test_accuracy = [[], [], [], []]

def SVMs(kernel="linear",decision_function_shape = "ovo",C = 1,gamma = "scale",x_data = [],y_data = [],degree = 3,coef0 = 0,max_iter = -1,probability = False,shrinking = True,cache_size = 200 , tol=0.001 , verbose = False,class_weight = None):
    start = time.time()
    if decision_function_shape == "ovo":
        svms = svm.SVC(kernel=kernel, C=C, decision_function_shape="ovo",gamma=gamma,degree=degree,coef0=coef0,tol=tol,cache_size=cache_size,verbose=verbose,max_iter=max_iter,probability=probability,shrinking=shrinking,class_weight=class_weight).fit(x_data, y_data)
    else:
        svms = OneVsRestClassifier(svm.SVC(kernel=kernel, C=C,gamma=gamma,degree=degree,coef0=coef0,tol=tol,cache_size=cache_size,verbose=verbose,max_iter=max_iter,probability=probability,shrinking=shrinking,class_weight=class_weight)).fit(x_data, y_data)
    stop = time.time()
    return svms,(stop-start)

#print the results
def show_results(train_times,train_accuracy,test_accuracy,kernel_names):
    for i in range(len(kernel_names)):
        print("-->For SVM using kernel: " + str(kernel_names[i]))
        print("Train time: "+str(train_times[i][0]))
        print("Train accuracy: "+str(train_accuracy[i][0]))
        print("Test accuracy: "+str(test_accuracy[i][0]))

#print the results if k cross fold validation was used
def show_folds_results(train_times,train_accuracy,test_accuracy,kernel_names,N):
    N = list(range(1, N+1))
    ta = [[],[],[]]
    for i in range(len(kernel_names)):
        print("--------------------------------------------")
        print("For SVM using kernel: "+str(kernel_names[i]))
        print("Train time for each fold: "+str(train_times))
        print("Average train time: "+str(helper_methods.calc_list_average(train_times[i])))
        print("Train accuracy for each fold: "+str(train_accuracy))
        print("Average train accuracy: "+str(helper_methods.calc_list_average(train_accuracy[i])))
        print("Test accuracy for each fold: "+str(test_accuracy))
        print("Average test accuracy: "+str(helper_methods.calc_list_average(test_accuracy[i])))

        ta[0].append(helper_methods.calc_list_average(train_times[i]))
        ta[1].append(helper_methods.calc_list_average(train_accuracy[i]))
        ta[2].append(helper_methods.calc_list_average(test_accuracy[i]))

        plt.subplot(2, 2, i + 1)  # i+1 is the index
        # plot lines
        plt.plot(N,train_times[i], label="Train Time")
        plt.plot(N,train_accuracy[i], label="Train Accuracy")
        plt.plot(N,test_accuracy[i], label="Test Accuracy")
        plt.legend()
    #plt.show()

    stats_data = np.array(ta)
    stats = pd.DataFrame(stats_data, columns=['linear', 'rbf', 'poly', 'sigmoid'],
                         index=['time', 'train accuracy', 'test accuracy'])
    print(stats)


#plot the svm prediction
def plot_svms(X,y,clf,fig_title):
    fig = plt.figure()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = helper_methods.make_meshgrid(X0, X1)

    for i in range(len(clf)):
        plt.subplot(2, 2, i + 1) #i+1 is the index
        helper_methods.plot_contours(plt, clf[i], xx, yy, cmap=plt.cm.coolwarm, alpha=1)
        plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=15, edgecolors='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(kernel_names[i])

    fig.suptitle(fig_title)
    plt.show()

#Calculate accuracy for each kernel per class
def per_class_accuracy(x_test,y_test,cf1,cf2,cf3,cf4,n_classes):
    classes = []
    result = []
    for i in range(n_classes):
        classes.append(i+1)
        xnew = []
        ynew = []
        for j in range(len(x_test)):
            if y_test[j] == i+1:
                ynew.append(y_test[j])
                xnew.append(x_test[j])
        result.append([cf1.score(xnew, ynew),cf2.score(xnew, ynew),cf3.score(xnew, ynew),cf4.score(xnew, ynew)])
    stats = pd.DataFrame(result, columns=['linear', 'rbf', 'poly', 'sigmoid'],index=classes)
    print(stats)

#train the svm model for each kernel , keep statistics (train time, accuracy etc) and show them
def run_model(x_train,y_train,x_test,y_test):
    # LINEAR KERNEL SVM
    linear_kernel_svm, train_time = SVMs( kernel='linear',x_data=x_train,y_data=y_train)
    train_times[0].append(train_time)
    train_accuracy[0].append(linear_kernel_svm.score(x_train, y_train))
    test_accuracy[0].append(linear_kernel_svm.score(x_test, y_test))

    # RBF KERNEL SVM
    rbf_kernel_svm, train_time = SVMs(kernel="rbf", x_data=x_train,y_data=y_train)
    train_times[1].append(train_time)
    train_accuracy[1].append(rbf_kernel_svm.score(x_train, y_train))
    test_accuracy[1].append(rbf_kernel_svm.score(x_test, y_test))

    # POLYNOMIAL KERNEL SVM
    poly_kernel_svm, train_time = SVMs(kernel="poly", x_data=x_train,y_data=y_train)
    train_times[2].append(train_time)
    train_accuracy[2].append(poly_kernel_svm.score(x_train, y_train))
    test_accuracy[2].append(poly_kernel_svm.score(x_test, y_test))

    # SIGMOID KERNEL SVM
    sigmoid_kernel_svm, train_time = SVMs(kernel='sigmoid', x_data=x_train,y_data=y_train)
    train_times[3].append(train_time)
    train_accuracy[3].append(sigmoid_kernel_svm.score(x_train, y_train))
    test_accuracy[3].append(sigmoid_kernel_svm.score(x_test, y_test))

    stats_data = np.array([(train_times[0][-1], train_times[1][-1], train_times[2][-1], train_times[3][-1]),
                     (train_accuracy[0][-1], train_accuracy[1][-1], train_accuracy[2][-1], train_accuracy[3][-1]),
                     (test_accuracy[0][-1], test_accuracy[1][-1], test_accuracy[2][-1], test_accuracy[3][-1])])
    stats = pd.DataFrame(stats_data, columns=['linear', 'rbf', 'poly', 'sigmoid'],
                       index=['time', 'train accuracy', 'test accuracy'])
    print(stats)

    #Accuracy Per Class
    per_class_accuracy(x_test,y_test,linear_kernel_svm,rbf_kernel_svm,poly_kernel_svm,sigmoid_kernel_svm,max(y_test))

    #plot_svms(x_train, y_train, [linear_kernel_svm,rbf_kernel_svm,poly_kernel_svm,sigmoid_kernel_svm],"Train Data")
    #plot_svms(x_test, y_test, [linear_kernel_svm,rbf_kernel_svm,poly_kernel_svm,sigmoid_kernel_svm],"Test Data")

#k cross fold validation technique if test and train datasets are not predefined
def K_Fold_Validation(data,labels,N):
    data_array = np.array(data)
    labels_array = np.array(labels)
    kf = KFold(n_splits=N)
    for train_index, test_index in kf.split(data):
        train_data_x, test_data_x = data_array[train_index],data_array[test_index]
        train_data_y, test_data_y = labels_array[train_index],labels_array[test_index]
        run_model(train_data_x,train_data_y,test_data_x,test_data_y)

    show_folds_results(train_times,train_accuracy,test_accuracy,kernel_names,N)
#X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .75)

def optimize_parameters(x_data,y_data,param_grid):
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3,cv=5,n_jobs=-1)
    grid.fit(x_data, y_data)
    print(grid.best_params_)
    print(grid.best_estimator_)

#shuttle dataset
x_train,y_train = data_preprocessing.get_shuttle_dataset("G:\\shuttle_train..txt")
x_test,y_test = data_preprocessing.get_shuttle_dataset("G:\\shuttle_test.txt")
run_model(x_train,y_train,x_test,y_test)

#iris dataset
#x,y = data_preprocessing.get_iris_dataset("G:\\iris.data.csv")
#K_Fold_Validation(x,y,10)

#eeg emotions dataset
#x,y = data_preprocessing.get_eeg_biosignals("G:\\emotions.csv")
#K_Fold_Validation(x,y,5)

#For testing model parameters combinations
param_grid_linear = {'C': [0.01,0.09,0.1,0.2,0.5],
              'gamma': [10,2, 1, 0.1],
              'tol': [0.001, 0.01, 0.1, 0.5, 1],
              'kernel': ['linear']}
param_grid_rbf = {'C': [8,9,10,11,12],
              'gamma': [0.0008,0.001,0.005],
              'tol': [0.08, 0.1, 0.2, 0.3],
              'degree':[1,2,3],
              'kernel': ['rbf']}
param_grid_sigmoid = {'C': [0.1,0.2,0.3,0.4,0.5],
              'gamma': [0.005,0.001,0.0001],
              'coef0':[0],
              'degree':[1,2,3],
              'kernel': ['sigmoid']}
param_grid_poly = {'C': [0.1, 1, 5, 10],
              'gamma': [10,2, 1, 0.1, 0.001],
              'tol': [0.001, 0.01, 0.1, 0.5],
              'coef0':[0,0.1,0.5,1],
              'degree':[1,2,3],
              'kernel': ['poly']}
#optimize_parameters(x,y,param_grid_poly)
