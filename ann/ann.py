#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''

@author: yangfengguang

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@contact: yangbisheng2009@gmail.com

@file: neuralnet.py

@time: 2018/2/3 15:37

@desc:

'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os

import numpy as np
# import matplotlib as plt
import sklearn
import cPickle

home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
bin_dir = os.path.join(home_dir, 'bin')
model_dir = os.path.join(home_dir, 'model')
conf_dir = os.path.join(home_dir, 'conf')
data_dir = os.path.join(home_dir, 'data')
lib_dir = os.path.join(home_dir, 'lib')
log_dir = os.path.join(home_dir, 'log')
sys.path.append(lib_dir)

import gen_data

def lr(X, y):
    # Train the logistic rgeression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)

    # Plot the decision boundary
    # plot_decision_boundary(lambda x: clf.predict(x))
    # plt.title("Logistic Regression")

class ANN(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.model = None
        self.model_path = os.path.join(model_dir, 'ann.model')
        self.num_examples = 0 # training set size
        self.nn_input_dim = 2  # input layer dimensionality
        self.nn_output_dim = 2  # output layer dimensionality

        # Gradient descent parameters (I picked these by hand)
        self.epsilon = 0.01  # learning rate for gradient descent
        self.reg_lambda = 0.01  # regularization strength
        self.load()

    def load(self):
        self.X, self.y = gen_data.gen_2dim_data()
        self.num_examples = len(self.X)
        # print self.X, self.y

    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self, model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = self.X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    def predict(self, x=np.array([1.5215205, -0.1258923])):
        """ Helper function to predict an output (0 or 1) """
        model = cPickle.load(open(self.model_path, 'rb'))
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        expect_label = np.argmax(probs, axis=1)
        return expect_label

    def build_model(self, nn_hdim=5, num_passes=20000, print_loss=False):
        """
        This function learns parameters for the neural network and returns the model.
        nn_hdim: Number of nodes in the hidden layer
        num_passes: Number of passes through the training data for gradient descent
        print_loss: If True, print the loss every 1000 iterations
        """
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))

        # This is what we return at the end
        self.model = {}

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):

            # Forward propagation
            z1 = self.X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(self.num_examples), self.y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(self.X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, self.calculate_loss(self.model))
        cPickle.dump(self.model, open(self.model_path, 'wb'))
        return self.model


if __name__ == '__main__':
    o = ANN()
    o.build_model(nn_hdim=4, num_passes=20000, print_loss=True)
    expect_label = o.predict(np.array([1.5215205, -0.1258923]))
    print expect_label
