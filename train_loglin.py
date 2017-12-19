import loglinear as ll
import random
import utils
import numpy as np

STUDENT={'name': 'Noa Yehezkel Lubin',
         'ID': '305097552'}

def feats_to_vec(features):
    """
    Calculates most common features histogram.
    features: list of features.
    """
    bigrams = utils.text_to_bigrams(features)
    feat_vec = np.zeros(len(utils.vocab))
    for b in bigrams:
        if b in utils.vocab:
            feat_vec[utils.F2I[b]] += 1
    return feat_vec/len(bigrams)

def accuracy_on_dataset(dataset, params):
    """
    Calculates accuracy by using loglinear predict.
    dataset: a list of (label, feature) pairs.
    params: list of parameters (initial values)
    """
    total = good = 0.0
    for label, features in dataset:
        total+=1
        predicted_label = ll.predict(feats_to_vec(features), params)
        if predicted_label == utils.L2I[label]:
            good += 1
    return float(good) / total

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.
    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            params[0] -= grads[0] * (learning_rate ** I)
            params[1] -= grads[1] * (learning_rate ** I)
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def get_tag(dataset, params):
    """
    Tgas the dataset based on trained params
    dataset: a list of (label, feature) pairs.
    params: list of parameters (initial values)
    """
    f = open('test.pred.ll', 'w')
    for label, features in dataset:
        predicted_label = ll.predict(feats_to_vec(features), params)
        f.write("%s\n" % utils.I2L[predicted_label])
    f.close()

if __name__ == '__main__':
    train_data = utils.read_data(r'train')
    dev_data = utils.read_data(r'dev')
    test_data = utils.read_data(r'test')
    in_dim = len(utils.vocab)
    out_dim = len(utils.L2I)
    params = ll.create_classifier(in_dim, out_dim)
    num_iterations = 200
    learning_rate = 0.99
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    #get_tag(test_data, trained_params)

