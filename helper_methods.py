import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay

#create the mesh
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

#put the prediction results into a color plot
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#return the average of a list containing numbers
def calc_list_average(ls):
    sum = 0
    for i in range(len(ls)):
        sum += ls[i]
    return sum/len(ls)

#print the results after the k-fold validation
def show_folds_result(train_times,train_accuracy,test_accuracy):
    print("-------------------------------------------------")
    print("Average train time: " + str(format(float(calc_list_average(train_times)),'f')))
    print("Train accuracy for each fold: " + str(train_accuracy))
    print("Average train accuracy: " + str(calc_list_average(train_accuracy)))
    print("Test accuracy for each fold: " + str(test_accuracy))
    print("Average test accuracy: " + str(calc_list_average(test_accuracy)))

#plot the model predictions given a model clf
def plot(X,y,clf,title):

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=plt.cm.coolwarm, ax=ax, response_method="predict"
    )

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k", s=20)
    plt.title(title)
    plt.show()

#Calculate accuracy per class for a given model (KNN , Nearest Centroid)
def per_class_accuracy(x,y,model,n_classes):
    classes = []
    result = []
    for i in range(n_classes):
        classes.append(i+1)
        xnew = []
        ynew = []
        for j in range(len(x)):
            if y[j] == i+1:
                ynew.append(y[j])
                xnew.append(x[j])
        result.append([model.score(xnew, ynew)])
    stats = pd.DataFrame(result, columns=['model'],index=classes)
    print(stats)