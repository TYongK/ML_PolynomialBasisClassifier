#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt


# In[2]:


# Simple Python3 program to multiply two polynomials
# A[] represents coefficients of first polynomial
# B[] represents coefficients of second polynomial
# m and n are sizes of A[] and B[] respectively
def multiply(A, B, m, n):
    
    prod = [0.] * (m + n - 1);
    
    # Multiply two polynomials term by term
    # Take ever term of first polynomial
    
    for i in range(m):
        # Multiply the current term of first
        # polynomial with every term of
        # second polynomial.
        for j in range(n):
            #Error Location 20220203_Start  
            prod[i + j] += A.T[i] * B.T[j];
            #Error Location 20220203_End
    return prod;
 
def multiplyVTwo(A, B, m, n):
    
    prod = np.zeros(((m + n + 1),1));
    #prod = [0.] * (m + n - 1);
    
    # Multiply two polynomials term by term
    # Take ever term of first polynomial
    
    prod[0, 0] = A.T[0]
    prod[1, 0] = B.T[0]
    
    for i in range(m):
        # Multiply the current term of first
        # polynomial with every term of
        # second polynomial.
        for j in range(n):
 
            prod[i + j + 2, 0] += A.T[i] * B.T[j]

    return prod;
            
#Reference: https://www.geeksforgeeks.org/multiply-two-polynomials-2/


# In[26]:


# opening the 'my_csv' file to read its contents
with open('E:\\_Github\\ML_PolynomialBasisClassifier\\RealEstateDataSet003.csv', newline = '', encoding="utf-8-sig") as file:
    reader = csv.reader(file,
                        quoting = csv.QUOTE_ALL,
                        delimiter = ' ')
     
    # storing all the rows in an output list
    output = []
    for row in reader:
        output.append(row[:][0].split(","))
        
dataset_string = np.asarray(output[1:])
dataset_string[dataset_string == ''] = '0'
dataset = dataset_string.astype(float)

def DataOrganize(Dataset):
    dataset_features = Dataset[:,0:-1]
    dataset_labels = Dataset[:,-1][np.newaxis].T
    return dataset_features, dataset_labels

def SaveGraph2DAnimationImages (Dataset, Result_th, Result_thZero, Order, XOne_Coluumn_Int, XTwo_Coluumn_Int, iteration):
    
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    xOne_Data = dataset_features[:, XOne_Coluumn_Int:XOne_Coluumn_Int+1]
    xTwo_Data = dataset_features[:, XTwo_Coluumn_Int:XTwo_Coluumn_Int+1]    
    dataset_selected = np.concatenate((xOne_Data,xTwo_Data),axis = 1)
    
    positive_features = dataset_selected[np.where(dataset_labels[:,0] == 1)]
    negative_features = dataset_selected[np.where(dataset_labels[:,0] == -1)]
    
    PredResolution = 10 #should be Natural Number.
    PredxOne = int((np.amax(xOne_Data) - np.amin(xOne_Data) + 4) * PredResolution)
    PredxTwo = int((np.amax(xTwo_Data) - np.amin(xTwo_Data) + 4) * PredResolution)            
    PredBackground = np.zeros((PredxOne, PredxTwo))
    
    for i in range(PredxOne):
        for j in range(PredxTwo):
            
                #Constructing Polynomial Basis_Start
                xOne_poly = np.full((1,Order), np.amin(xOne_Data) -2 + i / PredResolution)
                xTwo_poly = np.full((1,Order), np.amin(xTwo_Data) -2 + j / PredResolution)
                expo_array = np.arange(1, xOne_poly.size + 1, 1, dtype=int)
            
                xOne_poly = np.power(xOne_poly, expo_array)
                xTwo_poly = np.power(xTwo_poly, expo_array)
                
                #Error Location 20220203_Start
                poly_Basis = multiplyVTwo(xOne_poly, xTwo_poly, xOne_poly.size, xTwo_poly.size)
                #Error Location 20220203_End
                #Constructing Polynomial Basis_End
                
                PredBackground[i,j] = np.sign(np.dot(Result_th.T, poly_Basis) + Result_thZero)
                
    #for i in range(dataset_labels.shape[0]):
        #print(dataset_labels[i:i+1,:]*(np.dot(Result_th.T, poly_Basis) + Result_thZero))
                
    NegPred = np.array(np.where(PredBackground == -1.)).T
    NegPred = (NegPred/PredResolution + [np.amin(xOne_Data)-2, np.amin(xTwo_Data)-2])

    plt.clf()
    
    f = plt.figure()
    xOneLength = np.amax(xOne_Data) - np.amin(xOne_Data)+2
    xTwoLength = np.amax(xTwo_Data) - np.amin(xTwo_Data)+2   
    xOneLRatio = xOneLength/(xOneLength + xTwoLength)
    xTwoLRatio = xTwoLength/(xOneLength + xTwoLength)
    plotSizeMult = 20
    f.set_figwidth(xOneLRatio * plotSizeMult)
    f.set_figheight(xTwoLRatio * plotSizeMult)
    #Size of Plot_End
    
    plt.axis([np.amin(xOne_Data)-2, np.amax(xOne_Data)+2, np.amin(xTwo_Data)-2, np.amax(xTwo_Data)+2])
    plt.plot(NegPred[:,0], NegPred[:,1], '#9f9f9f', marker="s", linestyle='none', markersize= 3000/PredResolution/72)
    plt.plot(positive_features[:,0], positive_features[:,1], 'ro', markersize=5)
    plt.plot(negative_features[:,0], negative_features[:,1], 'bo', markersize=5)
    plt.savefig("E:\_Github\ML_PolynomialBasisClassifier\images\PcResult"+ str(iteration) +'.png')
    
def ShowGraph2D (Dataset, Result_th, Result_thZero, Order, XOne_Coluumn_Int, XTwo_Coluumn_Int):

    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    xOne_Data = dataset_features[:, XOne_Coluumn_Int:XOne_Coluumn_Int+1]
    xTwo_Data = dataset_features[:, XTwo_Coluumn_Int:XTwo_Coluumn_Int+1]    
    dataset_selected = np.concatenate((xOne_Data,xTwo_Data),axis = 1)
    
    positive_features = dataset_selected[np.where(dataset_labels[:,0] == 1)]
    negative_features = dataset_selected[np.where(dataset_labels[:,0] == -1)]
    
    PredResolution = 10 #should be Natural Number.
    PredxOne = int((np.amax(xOne_Data) - np.amin(xOne_Data) + 4) * PredResolution)
    PredxTwo = int((np.amax(xTwo_Data) - np.amin(xTwo_Data) + 4) * PredResolution)            
    PredBackground = np.zeros((PredxOne, PredxTwo))
    
    for i in range(PredxOne):
        for j in range(PredxTwo):
            
                #Constructing Polynomial Basis_Start
                xOne_poly = np.full((1,Order), np.amin(xOne_Data) -2 + i / PredResolution)
                xTwo_poly = np.full((1,Order), np.amin(xTwo_Data) -2 + j / PredResolution)
                expo_array = np.arange(1, xOne_poly.size + 1, 1, dtype=int)
            
                xOne_poly = np.power(xOne_poly, expo_array)
                xTwo_poly = np.power(xTwo_poly, expo_array)
                
                #Error Location 20220203_Start
                poly_Basis = multiplyVTwo(xOne_poly, xTwo_poly, xOne_poly.size, xTwo_poly.size)
                #Error Location 20220203_End
                #Constructing Polynomial Basis_End
                
                PredBackground[i,j] = np.sign(np.dot(Result_th.T, poly_Basis) + Result_thZero)
                
    #for i in range(dataset_labels.shape[0]):
        #print(dataset_labels[i:i+1,:]*(np.dot(Result_th.T, poly_Basis) + Result_thZero))
                
    NegPred = np.array(np.where(PredBackground == -1.)).T
    NegPred = (NegPred/PredResolution + [np.amin(xOne_Data)-2, np.amin(xTwo_Data)-2])

    plt.clf()
    
    f = plt.figure()
    xOneLength = np.amax(xOne_Data) - np.amin(xOne_Data)+2
    xTwoLength = np.amax(xTwo_Data) - np.amin(xTwo_Data)+2   
    xOneLRatio = xOneLength/(xOneLength + xTwoLength)
    xTwoLRatio = xTwoLength/(xOneLength + xTwoLength)
    plotSizeMult = 20
    f.set_figwidth(xOneLRatio * plotSizeMult)
    f.set_figheight(xTwoLRatio * plotSizeMult)
    #Size of Plot_End
    
    plt.axis([np.amin(xOne_Data)-2, np.amax(xOne_Data)+2, np.amin(xTwo_Data)-2, np.amax(xTwo_Data)+2])
    plt.plot(NegPred[:,0], NegPred[:,1], '#9f9f9f', marker="s", linestyle='none', markersize= 3000/PredResolution/72)
    plt.plot(positive_features[:,0], positive_features[:,1], 'ro', markersize=5)
    plt.plot(negative_features[:,0], negative_features[:,1], 'bo', markersize=5)
    plt.show()
    
    
def Perceptron (Dataset, iteration, Order, XOne_Coluumn_Int, XTwo_Coluumn_Int, SaveImages):   
    
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    xOne_Data = dataset_features[:, XOne_Coluumn_Int:XOne_Coluumn_Int+1]
    xTwo_Data = dataset_features[:, XTwo_Coluumn_Int:XTwo_Coluumn_Int+1]   
    dataset_selected = np.concatenate((xOne_Data,xTwo_Data),axis=1)
    
    A = np.zeros(Order)
    B = np.zeros(Order)
    
    th = np.asarray(multiplyVTwo(A, B, A.size, B.size))
    thZero = 0
    
    for t in range(iteration):
        changed = False
        #print()
        for i in range(dataset_labels.shape[0]):
            
            xOne_poly = np.full((1,Order), dataset_selected[i,0])
            xTwo_poly = np.full((1,Order), dataset_selected[i,1])
            expo_array = np.arange(1, xOne_poly.size+1, 1, dtype=int)
        
            xOne_poly = np.power(xOne_poly, expo_array)
            xTwo_poly = np.power(xTwo_poly, expo_array)
            
            poly_Basis = np.asarray(multiplyVTwo(xOne_poly, xTwo_poly, xOne_poly.size, xTwo_poly.size))

            #print(np.dot(th.T, poly_Basis) + thZero)
            if dataset_labels[i:i+1,:]*(np.dot(th.T, poly_Basis) + thZero) <= 0.:
                th = th + poly_Basis*dataset_labels[i:i+1,:]
                thZero = thZero + dataset_labels[i:i+1,:]
                if SaveImages == True:
                    SaveGraph2DAnimationImages(dataset, th, thZero,  order, 7, 12, (t+1)*(i+1))
                changed = True
                
        if changed == False:
            print("Found Seperation")
            break
    return th, thZero

order = 2
result_th, result_thZero = Perceptron(dataset, 60000, order, 7, 12, True)
#ShowGraph2D(dataset, result_th, result_thZero, order, 7, 12)


# In[ ]:





# In[ ]:




