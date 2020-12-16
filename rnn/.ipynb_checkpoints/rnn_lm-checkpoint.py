#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import csv
import datetime
import itertools
import cPickle
import operator
import time

import nltk
import numpy as np

home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
bin_dir = os.path.join(home_dir, 'bin')
model_dir = os.path.join(home_dir, 'model')
conf_dir = os.path.join(home_dir, 'conf')
data_dir = os.path.join(home_dir, 'data')
lib_dir = os.path.join(home_dir, 'lib')
log_dir = os.path.join(home_dir, 'log')

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim),\
                                   (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),\
                                   (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),\
                                   (hidden_dim, hidden_dim))
        self.x_train = os.path.join(data_dir, 'x_train.pkl')
        self.y_train = os.path.join(data_dir, 'y_train.pkl')

        self.index_to_word = []
        self.word_to_index = {}

    def gen_xy(self):
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading CSV file..."
        with open('../data/reddit-comments-2015-08.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(
                *[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in
                         sentences]
        print "Parsed %d sentences." % (len(sentences))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])

        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (
            vocab[-1][0], vocab[-1][1])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        print "\nExample sentence: '%s'" % sentences[0]
        print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

        # Create the training data
        X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        # Print an training data example

        x_example, y_example = X_train[17], y_train[17]
        print "x:\n%s\n%s" % (" ".join([self.index_to_word[x] for x in x_example]), x_example)
        print "\ny:\n%s\n%s" % (" ".join([self.index_to_word[x] for x in y_example]), y_example)
        cPickle.dump(X_train, open(self.x_train, 'wb'))
        cPickle.dump(y_train, open(self.y_train, 'wb'))
        print 'gen xy done.'

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        # o = 45*8000, the probality of each word in each position.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            # U[:,x[t]] = 100*1, W.dot(s[t-1]) = 100*100 * 1*100
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            # V.dot(s[t]) = 8000*100 * 1*100
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            #print len(y[i])
            #print o[np.arange(len(y[i])), y[i]].shape
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        # N is the total number of all the words
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

    def generate_sentence(self):
        # We start the sentence with the start token
        new_sentence = [self.word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == self.word_to_index[sentence_end_token]:
            next_word_probs = self.forward_propagation(new_sentence)
            sampled_word = self.word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == self.word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [self.index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str

if __name__ == '__main__':
    np.random.seed(10)
    obj = RNNNumpy(vocabulary_size)
    obj.gen_xy()
    # x and y make some difference which is one has 'START' another has the 'END'.
    x_train = cPickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
    y_train = cPickle.load(open(os.path.join(data_dir, 'y_train.pkl'), 'rb'))

    # Just test predict function.
    predictions = obj.predict(x_train[10])

    # Limit to 1000 examples to save time.
    print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
    print "Actual loss: %f" % obj.calculate_loss(x_train[:1000], y_train[:1000])

    # Test gradient check.
    grad_check_vocab_size = 100
    np.random.seed(10)
    model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])

    # Test how long dose one step cost.
    start = time.time()
    obj.numpy_sdg_step(x_train[10], y_train[10], 0.005)
    end = time.time()
    print 'one step consume: %f' % (end - start)

    # train
    losses = obj.train_with_sgd(x_train, y_train, nepoch=1, evaluate_loss_after=1)

    '''
    # Generate sentences.
    num_sentences = 10
    senten_min_length = 7
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = obj.generate_sentence()
        print " ".join(sent)
    '''