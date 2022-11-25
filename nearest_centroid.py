import numpy as np
from sklearn.model_selection import KFold
import data_preprocessing
import helper_methods
from sklearn.neighbors import NearestCentroid
import time

train_times = []
train_accuracy = []
test_accuracy = []

#train the model and calculate the train time
def train_model(x_train,y_train,metric,threshhold):
    start = time.time()
    model = NearestCentroid(metric=metric,shrink_threshold=threshhold).fit(x_train, y_train)
    stop = time.time()
    return model,(stop-start)

#train the model, calculate the prediction scores for train,test and show the results
def run(x_train,y_train,x_test,y_test):
    m,t = train_model(x_train,y_train,"euclidean",None)
    train_accuracy.append(m.score(x_train, y_train))
    test_accuracy.append(m.score(x_test, y_test))
    train_times.append(t)

    print("Train time: "+str(train_times[-1]))
    print("Train accuracy: "+str(train_accuracy[-1]))
    print("Test accuracy: "+str(test_accuracy[-1]))

    helper_methods.per_class_accuracy(x_test,y_test,m,max(y_test))
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

#iris dataset
#x,y = data_preprocessing.get_iris_dataset("G:\\iris.data.csv")
#K_Fold_Validation(x,y,10)

#shuttle dataset
#x_train,y_train = data_preprocessing.get_shuttle_dataset("G:\\shuttle_train..txt")
#x_test,y_test = data_preprocessing.get_shuttle_dataset("G:\\shuttle_test.txt")
#run(np.array(x_train),np.array(y_train), np.array(x_test), np.array(y_test))

#eeg emotions detection dataset
x,y = data_preprocessing.get_eeg_biosignals("G:\\emotions.csv")
K_Fold_Validation(x,y,5)