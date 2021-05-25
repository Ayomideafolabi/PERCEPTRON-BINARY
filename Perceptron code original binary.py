# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 02:58:09 2021

@author: ayomy
"""
# Data preparation and loading


import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    
    def __init__(self, features_number,learn_rate, maxim_iteration):
        self.weights = np.zeros(1 + features_number) #initialize weights to zero
        self.maxim_iteration = maxim_iteration
        self.learn_rate = learn_rate
    
    # The function trains the model returning the train weights
    def fit(self,X_train,y_train):
        for i in range(self.maxim_iteration):
            for inputs,label in list(zip(X_train,y_train)):
                total = self.weights[0] + np.dot(inputs,self.weights[1:])
                if total > 0:
                    y_predicted = 1
                else:
                    y_predicted = -1
                self.weights[0] = self.weights[0] + (self.learn_rate *(label-y_predicted))
                self.weights[1:] = self.weights[1:]+(inputs*self.learn_rate *(label-y_predicted))
        return self.weights
                                          
        
    # This function predict all the final label   
    def predict(self,X_test): 
       y_predict_final = np.zeros(len(X_test)) 
       for i in range(len(X_test)):
            total = self.weights[0] + np.dot(X_test[i],self.weights[1:])
            if total > 0:
               y_predict_final[i] = 1
            else:
               y_predict_final[i] = -1
           
       return y_predict_final
   
 
    # This function compute the prediction accuracy                                      
    def prediction_accuracy(self,X_test,y_test):
        correctcount = 0
        wrongcount = 0
        y_predict_final = self.predict(X_test)
        testlabel_and_predictedlabel = list(zip(y_test,y_predict_final))
        for i in range(len(testlabel_and_predictedlabel)):
           if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
              correctcount += 1
           else:
              wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio
    
    # This computes the normalized weights between 0 and 1 to show the predictor importance
    def weights_relativeimportance(self,X_train,y_train):
        full_weights = self.fit(X_train,y_train)
        full_weights = np.abs(full_weights) # absolute weights since direction does not matter
        weightsimportance = full_weights[1:].tolist()
        Normalized_weights = []
        
        for i in range(len(weightsimportance)):
            minweight = min(weightsimportance)
            maxweight = max(weightsimportance)
            b = (weightsimportance[i] - minweight)/(maxweight- minweight)
            Normalized_weights.append(b)    
        Normalized_weights_index = np.argsort(Normalized_weights)
        Normalized_weights_index = Normalized_weights_index[::-1][0:20]
        Normalized_weights = np.sort(Normalized_weights)
        Normalized_weights =  Normalized_weights[::-1][0:20]
        Feature_number = ["Feature "+str(i+1) for i in Normalized_weights_index]
        return Feature_number, Normalized_weights.tolist()
    
    # This function shows the plot
    def feature_importance_plot(self,X_train,y_train):
        Features = self.weights_relativeimportance(X_train,y_train)[0]
        Normalized_weights_20 =  self.weights_relativeimportance(X_train,y_train)[1]
        plt.bar(Features,Normalized_weights_20)
        plt.title("BINARY CLASSIFIER ATTRIBUTES IMPORTANCE BAR CHART NORMALIZED")
        plt.xlabel("Feature numbers")
        plt.ylabel("Importance level")
        plt.xticks(rotation =90)
        return plt.show()


import numpy as np
np.random.seed (0)                             
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
X, y = load_svmlight_file("C:/Users/ayomy/Documents/ML Mini Project/datatrain1.txt")
X_test, y_test = load_svmlight_file("C:/Users/ayomy/Documents/ML Mini Project/datatest.txt")
X_train,X_test,y_train,y_test = train_test_split(X.toarray(),y,test_size = 0.2)        
features_number = 122;learn_rate =0.1;maxim_iteration = 1       
Bin =Perceptron(features_number,learn_rate, maxim_iteration) 
Bin.fit(X_train,y_train) #train the system
print(Bin.prediction_accuracy(X_test,y_test)) # computes the prediction accuracy
print(Bin.feature_importance_plot(X_train,y_train)) # plot the charts
   
            
        
    
    
    

