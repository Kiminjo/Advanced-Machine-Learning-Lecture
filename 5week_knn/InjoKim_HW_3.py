# -*- coding: utf-8 -*-

# DO NOT CHANGE
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt




#%%

k_list = [3, 5, 7, 9, 11]

#%%
def wkNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement weighted kNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    
    y_pred = []
    for point in Xts :
        distance = pairwise_distances([point], Xtr)[0]
        index_within_k = np.argsort(distance)[0:k]
        
        if len(np.unique(ytr)) > 2 :
            weight = {0:0, 1:0, 2:0}
        else :
            weight = {0:0, 1:0}

        for index in index_within_k:
            if index == index_within_k[0] :
                weight[ytr[index]] = 1
                
            else :
                weight[ytr[index]] = (distance[index_within_k[-1]]-distance[index])/(distance[index_within_k[-1]]-distance[index_within_k[0]]) + weight[ytr[index]] 
        
        y_pred.append(max(weight, key=weight.get))    
        
    return y_pred

#%%
def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    
    y_pred = []
    point_per_class = {}
    
    if len(np.unique(ytr)) > 2 :
        class_num = 3
    else :
        class_num = 2
        
    for class_ in range(0,class_num) :
        point_per_class[class_] = Xtr[ytr==class_]
        
    for point in Xts :
        weight = {}
        for class_ in point_per_class :
            weight[class_] = np.sort(pairwise_distances(point_per_class[class_], [point]).reshape(1,-1)[0])[0:k]
            for i in range(1, k+1) :
                weight[class_][i-1] = weight[class_][i-1]/i
        
        sum_of_weight_per_class = {}
        for class_ in point_per_class:
            sum_of_weight_per_class[class_] = sum(weight[class_])
            
        y_pred.append(min(sum_of_weight_per_class, key=sum_of_weight_per_class.get))

    return y_pred


#%%

def accuracy(pred, test, accuracy) :
    
    if k == 3 and len(accuracy) > 0 :
        accuracy = []

    total_length = len(pred)
    number_of_same_element = sum(pred==test)
    accuracy.append(round(number_of_same_element/total_length, 3))

    return accuracy


#%%


def print_table(k, wknn_accuracy, pnn_accuracy, time) :
    
    print("Elapsed time : {}".format(time))
    print("------------------------------------")
    print("      k            wkNN             PNN          ")
    for i in range(len(k)) :
        print("    ", k[i], "        ", wknn_accuracy[i], "            ", pnn_accuracy[i])
    print("------------------------------------")
        

#%%

def plotting(k, wknn_pred, pnn_pred, Xts, yts, Xtr, ytr, X1, X2, mode) :
    plt.rcParams["figure.figsize"] = (10, 10)
    
    wk_TF = (wknn_pred == yts)
    p_TF = (pnn_pred == yts)
    
    val_Xtr_x = []
    val_Xtr_y = []
    val_Xts_x = []
    val_Xts_y = []
    wk_TF_x = []
    wk_TF_y = []
    p_TF_x = []
    p_TF_y = []
    
    point_per_train_class = {}
    point_per_test_class = {}
    
    for class_ in range(0,3) :
        point_per_train_class[class_] = Xtr[ytr==class_]
        point_per_test_class[class_] = Xts[yts==class_]
        
    if mode == 1:
        for val1, val2 in Xtr:
            val_Xtr_x.append(val1)
            val_Xtr_y.append(val2)
        for val1, val2 in Xts:
            val_Xts_x.append(val1)
            val_Xts_y.append(val2)
            
    if mode == 2 :    
        for val1, val2, val3, val4, val5, val6 in Xtr:
            val_Xtr_x.append(val1)
            val_Xtr_y.append(val2)
        for val1, val2, val3, val4, val5, val6 in Xts:
            val_Xts_x.append(val1)
            val_Xts_y.append(val2)
        
    for idx in range(len(wk_TF)):
        if wk_TF[idx] == True:
            continue
        else:
            wk_TF_x.append(val_Xtr_x[idx])
            wk_TF_y.append(val_Xtr_y[idx])
    for idx in range(len(p_TF2)):
        if p_TF2[idx] == True:
            continue
        else:
            p_TF_x.append(val_Xtr_x[idx])
            p_TF_y.append(val_Xtr_y[idx])
            
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(val_Xtr_x,val_Xtr2_y,c='blue',marker='o',s=18)
    plt.scatter(val_Xts_x,val_Xts2_y,c='blue',marker='x',s=18)
    plt.scatter(point_per_train_class[0][: ,0], point_per_train_class[0][:,1], color='purple',marker='o',s=18)
    plt.scatter(point_per_train_class[1][: ,0], point_per_train_class[1][:,1], color='green',marker='o',s=18)
    plt.scatter(point_per_train_class[2][: ,0], point_per_train_class[2][:,1], color='yellow',marker='o',s=18)
                
    plt.scatter(point_per_test_class[0][: ,0], point_per_test_class[0][:,1], color='purple',marker='x',s=18)
    plt.scatter(point_per_test_class[1][: ,0], point_per_test_class[1][:,1], color='green',marker='x',s=18)
    plt.scatter(point_per_test_class[2][: ,0], point_per_test_class[2][:,1], color='yellow',marker='x',s=18)
    plt.scatter(wk_TF_x,wk_TF_y,c='salmon',marker='s',s=30)
    plt.scatter(p_TF_x,p_TF_y,c='royalblue',marker='d',s=30)
    plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'],loc='lower right')






#%%
X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot

wknn_accuracy = []
pnn_accuracy  = []

for k in k_list : 
    data_1_start = time.time()
    
    y_pred_wknn_1 = wkNN(Xtr1, ytr1, Xts1, k)
    y_pred_pnn_1 = PNN(Xtr1, ytr1, Xts1, k)
    
    data_1_time = time.time() - data_1_start
    
    accuracy_of_wknn_1 = accuracy(y_pred_wknn_1, yts1, wknn_accuracy)
    accuracy_of_pnn_1 = accuracy(y_pred_pnn_1, yts1, pnn_accuracy)

print_table(k_list, accuracy_of_wknn_1, accuracy_of_pnn_1, data_1_time)
plotting(7, accuracy_of_wknn_1[2], accuracy_of_pnn_1[2], Xts1, yts1, Xtr1, ytr1, X1, X2, 1) 


#%%
X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot

wknn_accuracy = []
pnn_accuracy  = []

for k in k_list : 
    data_2_start = time.time()
    
    y_pred_wknn_2 = wkNN(Xtr2, ytr2, Xts2, k)
    y_pred_pnn_2 = PNN(Xtr2, ytr2, Xts2, k)
    
    data_2_time = time.time() - data_2_start
    
    accuracy_of_wknn_2 = accuracy(y_pred_wknn_2, yts2, wknn_accuracy)
    accuracy_of_pnn_2 = accuracy(y_pred_pnn_2, yts2, pnn_accuracy)

print_table(k_list, accuracy_of_wknn_2, accuracy_of_pnn_2, data_2_time)
plotting(7, accuracy_of_wknn_2[2], accuracy_of_pnn_2[2], Xts2, yts2, Xtr2, ytr2, X1, X2, 2) 

