# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 01:42:49 2018

@author: Shambhavi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:41:52 2018
@author: Shambhavi
"""
from mnist import MNIST #for data import 
import numpy as np
from os import listdir
import cv2
import os
from os.path import isfile, join
from scipy import ndimage
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from beautifultable import BeautifulTable
#Performing MLP Classifier
def mlpclassifier(dataset_rescaled,label,handwriting, label_hw):
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1) 
    percentage= []
    training= []
    validation=[]
    for i in range (70,80,5): 
        print (i)
        X_train, X_test, labels_train, labels_test = train_test_split(dataset_rescaled,label, test_size=i*0.01, random_state=0)
        mlp.fit(X_train, labels_train)
        print('Split Percentage: ',i,'%') 
        X_train_predict=mlp.predict(X_train)
        print ('Training accuracy using accuracy_score function',accuracy_score(labels_train,X_train_predict))
        X_test_predict=mlp.predict(handwriting)
        print(X_test_predict)
        print(label_hw)
        print ('Test accuracy using accuracy_score function',accuracy_score(label_hw,X_test_predict))
        percentage.append(i)
        training.append(accuracy_score(labels_train,X_train_predict))
        validation.append(accuracy_score(label_hw,X_test_predict))
        
#        Con_Matrix_train = confusion_matrix(labels_train,X_train_predict)
#        Con_Matrix_test = confusion_matrix(labels_test,X_test_predict)
#        print('Confusion Matrix for training data',Con_Matrix_train)
#        print('Confusion Matrix for test data',Con_Matrix_test)
    table = BeautifulTable()
    #method=["MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP","MLP"]
    method=["MLP","MLP","MLP"]
    table.column_headers = ["Spilt Percentage", "Method","Training Accuracy"]
    table.append_row([percentage,method, training])
    print(table)
    plt.plot(percentage,training,label='Training Accuracy')
    plt.plot (percentage,validation,label='Validation Accuracy')
    plt.title('Training and Validation accuracy plot using MLP Classifier')
    plt.xlabel('Split Percentage')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
#Performing Random Forest Classifier
#def rfclassifier(dataset_rescaled,label,handwriting, label_hw):
#    rfc= RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
#    percentage= []
#    training= []
#    validation=[]
#    for i in range (30,100,5): 
#        X_train, X_test, labels_train, labels_test = train_test_split(dataset_rescaled,label, train_size=i*0.01, random_state=0)
#        rfc.fit(X_train, labels_train)
#        print('Split Percentage: ',i,'%') 
#        X_train_predict=rfc.predict(X_train)
#        print ('Training accuracy using accuracy_score function',accuracy_score(labels_train,X_train_predict))
#        X_test_predict=rfc.predict(X_test)
#        print ('Test accuracy using accuracy_score function',accuracy_score(labels_test,X_test_predict))    
#        percentage.append(i)
#        training.append(accuracy_score(labels_train,X_train_predict))
#        validation.append(accuracy_score(labels_test,X_test_predict))
#    plt.plot(percentage,training,label='Training Accuracy')
#    plt.plot (percentage,validation,label='Validation Accuracy')
#    plt.title('Training and Validation accuracy plot using Random Forest Classifier')
#    plt.xlabel('Split Percentage')
#    plt.ylabel('Accuracy')
#    plt.legend(loc='best')
#    plt.show()
#    Con_Matrix_train = confusion_matrix(labels_train,X_train_predict)
#    Con_Matrix_test = confusion_matrix(labels_test,X_test_predict)
#    print('Confusion Matrix for training data',Con_Matrix_train)
#    print('Confusion Matrix for test data',Con_Matrix_test)
##Performing Stochastic Gradient Descent Classifier
#def sgdclassifier(dataset_rescaled,label,handwriting, label_hw):
#    clf = SGDClassifier(loss="hinge", penalty="l2")
#    percentage= []
#    training= []
#    validation=[]
#    for i in range (30,100,5): 
#        X_train, X_test, labels_train, labels_test = train_test_split(dataset_rescaled,label, train_size=i*0.01, random_state=0)
#        clf.fit(X_train, labels_train)
#        print('Split Percentage: ',i,'%') 
#        X_train_predict=clf.predict(X_train)
#        print ('Training accuracy using accuracy_score function',accuracy_score(labels_train,X_train_predict))
#        X_test_predict=clf.predict(X_test)
#        print ('Test accuracy using accuracy_score function',accuracy_score(labels_test,X_test_predict))
#        percentage.append(i)
#        training.append(accuracy_score(labels_train,X_train_predict))
#        validation.append(accuracy_score(labels_test,X_test_predict))
#    plt.plot(percentage,training,label='Training Accuracy')
#    plt.plot(percentage,validation,label='Validation Accuracy')
#    plt.title('Training and Validation accuracy plot using Stochastic Gradient Descent Classifier')
#    plt.xlabel('Split Percentage')
#    plt.ylabel('Accuracy')
#    plt.legend(loc='best')
#    plt.show()
#    Con_Matrix_train = confusion_matrix(labels_train,X_train_predict)
#    Con_Matrix_test = confusion_matrix(labels_test,X_test_predict)
#    print('Confusion Matrix for training data',Con_Matrix_train)
#    print('Confusion Matrix for test data',Con_Matrix_test)   
#Main 
def loadImages(path):

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        gray = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
        gray=cv2.bitwise_not(gray) 
        gray = cv2.resize(gray, (28, 28))
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)
        rows,cols = gray.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted
        loadedImages.append(gray)
        
    return loadedImages
        
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def pixellist(imgs):
    p=[]
    for i in range (0,100):
        pixel=np.array(imgs[i])
        images1=pixel.flatten()
        p.append(images1)
    return p
if __name__ == "__main__":
    mndata = MNIST('Samples')
    images, labels1 = mndata.load_training()
    images1, labels2 = mndata.load_testing()
    #Converting into numpy array
    x_train=np.array(images) #Training Data
    y_labels=np.array(labels1)#Training Data labels
    x_test=np.array(images1) #Test Data
    Y_labels=np.array(labels2)#Test Data labels
    #Stacking  up the training and test data set
    dataset= np.concatenate((x_train, x_test),axis=0)
    label= np.concatenate((y_labels, Y_labels),axis=0)
    #Rescaling the training data from 0 to 255
    dataset_rescaled= dataset/255
    path = 'images/'
    #images in an array
    imgs = loadImages(path)
    handw=pixellist(imgs)
    hd=np.array(handw)
    handwriting=hd/255
    #Labels
    label_list= [0,0,0,0,0,0,0,0,0,0,
           1,1,1,1,1,1,1,1,1,1,
           2,2,2,2,2,2,2,2,2,2,
           3,3,3,3,3,3,3,3,3,3,
           4,4,4,4,4,4,4,4,4,4,
           5,5,5,5,5,5,5,5,5,5,
           6,6,6,6,6,6,6,6,6,6,
           7,7,7,7,7,7,7,7,7,7,
           8,8,8,8,8,8,8,8,8,8,
           9,9,9,9,9,9,9,9,9,9]
    label_hw=np.array(label_list)
    mlpclassifier(dataset_rescaled,label,handwriting, label_hw)
    #    rfclassifier(dataset_rescaled,label)
#    sgdclassifier(dataset_rescaled,label)
   