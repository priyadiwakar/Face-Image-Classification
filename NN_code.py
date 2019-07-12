# -*- coding: utf-8 -*-
"""
@author: priya
"""

import tensorflow as tf
import numpy as np
import cv2
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import random
import math
import matplotlib.pyplot as plt


"""
Uncomment commented part to work with new dataset images from system 
instead of already saved in form of .npy files dataset

"""
#
#def data_collection(path_nface,path_face,N,D):
#    """
#    Prepare data from images 
#    path_nface = path where non face images are located
#    
#    path_face = path where face images are located
#    
#    N = number of images in your dataset for each class , in this case 12000
#    D=number of features, in this case 60*60*3 =10800
#       
#    """
#    
#    
#    t = np.zeros([N,D],dtype="float32")
#    u = np.zeros([N,D],dtype="float32")
#    
#    b=0
#    for img in glob.glob(str(path_nface/"*.jpg")):
#        
#        face= cv2.imread(img)
#        
#        t[b,:] = np.reshape(face, (1,D))
#        b+=1
#    
# 
#    a=0
#    for img in glob.glob(str(path_face/"*.jpg")):
#        nonface=cv2.imread(img)
#        u[a,:] = np.reshape(nonface, (1,D))
#        a+=1    
#    
#    return t,u
#
#path=Path("C:/Users/priya/Desktop/computervision/Project 2")
#    
#path_nface=path/"nonface"
#path_face = path/"face"
#N = 12000 #number of images in your dataset for each class , in this case 12000
#D=10800 #number of features, in this case 60*60*3 =10800
#T1=10000#number of training images for each class , in this case 10000
#T2=1000#number of testing images for each class , in this case 1000
#T3=1000#number of validation images for each class , in this case 1000
#indices = list(range(N))
#random.seed(4)
#random.shuffle(indices)
#"""
#Get the data using the function data_collection
#Also ,
#T1=number of training images for each class , in this case 10000
#T2=number of testing images for each class , in this case 1000
#T3=number of validation images for each class , in this case 1000
#
#"""
#total_nonfacedata,total_facedata =data_collection(path_nface,path_face,N,D)
#a=indices[:T1]
#b=indices[T1:T1+T2]
#c=indices[T1+T2:T1+T2+T3]
#
#y1=np.concatenate((np.zeros([T1,1]),np.ones([T1,1])),axis=0)
#y2=np.concatenate((np.ones([T1,1]),np.zeros([T1,1])),axis=0)
#y3=np.concatenate((y2,y1),axis=1)
#
#y4=np.concatenate((np.zeros([T2,1]),np.ones([T2,1])),axis=0)
#y5=np.concatenate((np.ones([T2,1]),np.zeros([T2,1])),axis=0)
#y6=np.concatenate((y5,y4),axis=1)
#
#y7=np.concatenate((np.zeros([T3,1]),np.ones([T3,1])),axis=0)
#y8=np.concatenate((np.ones([T3,1]),np.zeros([T3,1])),axis=0)
#y9=np.concatenate((y8,y7),axis=1)
#
##training data and labels, labels defined using one hot method
#train_data=np.concatenate((total_facedata[a],total_nonfacedata[a]))
#train_label=y3
#
##testing data and labels, labels defined using one hot method
#test_data=np.concatenate((total_facedata[b], total_nonfacedata[b]))
#test_label=y6
#
##validation data and labels, labels defined using one hot method
#valid_data=np.concatenate((total_facedata[c], total_nonfacedata[c]))
#valid_label=y9
#
#"""
#Save the data in .npy format for later use
#
#"""
#np.save('train_data.npy',train_data)
#np.save('train_label.npy',train_label)
#np.save('test_data.npy',test_data)
#np.save('test_label.npy',test_label)
#np.save('valid_data.npy',valid_data)
#np.save('valid_label.npy',valid_label)

"""Load already saved data"""
train_data=np.load('train_data.npy')
train_label=np.load('train_label.npy')

test_data=np.load('test_data.npy')
test_label=np.load('test_label.npy')

valid_data=np.load('valid_data.npy')
valid_label=np.load('valid_label.npy')


"""
Preprocess data to get zero centered and normalized data

"""
scaler = StandardScaler()
scaler.fit(train_data)
StandardScaler(copy=True, with_mean=True, with_std=True)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
valid_data = scaler.transform(valid_data)


num_nodes= 1000
batch_size = 100
beta = 0.01
num_labels=2
size=10800
graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder('float', shape=(batch_size, size))
    tf_train_labels = tf.placeholder('float', shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_data)
    tf_test_dataset = tf.constant(test_data)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer= tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
    # Loss function with L2 Regularization with beta=0.01
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(loss + beta * regularizers)
    
#    global_step = tf.Variable(0)  # count the number of steps taken.
#    start_learning_rate = 0.5
#    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 500, 0.5, staircase=True)
    # Optimizer.
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_2)
    
    # Predictions for validation 
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer= tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    
    valid_prediction = tf.nn.softmax(logits_2)
    
    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer= tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    
    test_prediction =  tf.nn.softmax(logits_2)
    
num_steps = 3001
losslist=[]
trainbatchacc=[]
validacc=[]
steps=[]

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
    
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_label.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_label[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        #lr = learning_rate.eval()

        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 3000 == 0):
            steps.append(step)
            losslist.append(l)
            ta=accuracy(predictions, batch_labels)
            trainbatchacc.append(ta)
            va=accuracy(valid_prediction.eval(), valid_label)
            validacc.append(va)
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(ta))
            print("Validation accuracy: {:.1f}".format(va))
    testacc=accuracy(test_prediction.eval(), test_label)
    print("Test accuracy: {:.1f}".format(testacc))
    plt.figure(1)
    plt.plot(steps,losslist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.show()
    plt.figure(2)
    p1,=plt.plot(steps,trainbatchacc,label="Training Accuracy")
    p2,=plt.plot(steps,validacc,label="Validation Accuracy", linestyle='--')
    plt.xlabel("Epoch")
    plt.legend([p1, p2],["Training Accuracy","Validation Accuracy"])
    plt.show()
