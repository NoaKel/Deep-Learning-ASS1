import numpy as np
import utils
import random
from grad_check import gradient_check


STUDENT={'name': 'Noa Yehezkel Lubin',
         'ID': '305097552'}

def feats_to_vec(features):
    bigrams = utils.text_to_bigrams(features)
    feature_vector = np.zeros(len(utils.vocab))
    for bigram in bigrams:
        if bigram in utils.vocab:
            feature_vector[utils.F2I[bigram]] += 1
    return feature_vector / len(bigrams)

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def accuracy_on_dataset(dataset, params):
    """
    Calculates accuracy by using loglinear predict.
    dataset: a list of (label, feature) pairs.
    params: list of parameters (initial values)
    """
    total = good = 0.0
    for label, features in dataset:
        total += 1
        predicted_label = predict(feats_to_vec(features), params)
        if predicted_label == utils.L2I[label]:
            good += 1
    return float(good) / total

def classifier_output(x, params):
    cur_function = x
    if len(params) == 0:
        return cur_function
    for i in range(len(params))[:-2:2]:
        cur_function = np.tanh(np.dot(cur_function, params[i]) + params[i+1])
    probs = softmax(np.dot(cur_function, params[-2]) + params[-1])

    return probs

def calc_cur_x(x, params):
    cur_function = x
    if len(params) == 0:
        return cur_function
    for i in range(len(params))[::2]:
        cur_function = np.tanh(np.dot(cur_function, params[i]) + params[i+1])
    return cur_function

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params, lambda_regul =0):
    y_gag = classifier_output(x, params)
    loss = -np.log(y_gag[y])
    for param in params:
        loss+= lambda_regul * (0.5 * np.sum(param**2))

    grads = []
    grads.append(y_gag)
    grads[0][y] = -(1 - y_gag[y])
    cur_x = calc_cur_x(x, params[:-2])
    grads.append(np.outer(cur_x, grads[0]))
    reverse_params = params[::-1]
    for i in range(len(params))[2::2]:
        cur_x = calc_cur_x(x, params[:-i-2])
        gb = np.dot(grads[i-2], reverse_params[i-1].T) * (np.array([1]*reverse_params[i].shape[0]) - np.tanh(np.dot(cur_x, reverse_params[i+1]) + reverse_params[i]) ** 2)
        gW = np.outer(cur_x, gb)
        grads += [gb, gW]

    grads.reverse()
    for i in range(len(params)):
        grads[i] += lambda_regul * params[i]
        
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    eps = -6 ** 0.5
    for dim in dims[:-1]:
        next_dim = dims[dims.index(dim) + 1]
        params.append(np.random.uniform(-eps / (dim + next_dim) ** 0.5, eps / (dim + next_dim) ** 0.5, (dim, next_dim)))
        params.append(np.random.uniform(-eps / next_dim ** 0.5, eps / next_dim ** 0.5, next_dim))
    return params

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        lambda_regul = 0.00005
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.L2I[label]  # convert the label to number if needed.

            loss, grads = loss_and_gradients(x, y, params,lambda_regul)
            cum_loss += loss
            for i in range(len(params)):
                params[i] -= grads[i] * (learning_rate ** I)

            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def grad_check_mlpn():
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1, 2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print test3
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

def get_tag(dataset, params):
    """
    Tgas the dataset based on trained params
    dataset: a list of (label, feature) pairs.
    params: list of parameters (initial values)
    """
    f = open('test.pred.mlpn', 'w')
    for label, features in dataset:
        predicted_label = predict(feats_to_vec(features), params)
        f.write("%s\n" % utils.I2L[predicted_label])
    f.close()

if __name__ == '__main__':
    # grad_check_mlpn()
    #
    # #   grad check
    # V, c, W, b, U, b_tag = create_classifier([3, 4, 5, 6])
    #
    # def _loss_and_V_grad(V):
    #     global b, b_tag, U, W, c
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[0]
    #
    # def _loss_and_c_grad(c):
    #     global b, b_tag, U, V, W
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[1]
    #
    #
    # def _loss_and_W_grad(W):
    #     global b, b_tag, U, V, c
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[2]
    #
    #
    # def _loss_and_b_grad(b):
    #     global W, b_tag, U, V, c
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[3]
    #
    #
    # def _loss_and_U_grad(U):
    #     global W, b, b_tag, V, c
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[4]
    #
    #
    # def _loss_and_b_tag_grad(b_tag):
    #     global W, U, b, V, c
    #     loss, grads = loss_and_gradients([1, 2, 3], 2, [V, c, W, b, U, b_tag])
    #     return loss, grads[5]
    #
    #
    # for _ in xrange(10):
    #     W = np.random.randn(W.shape[0], W.shape[1])
    #     b = np.random.randn(b.shape[0])
    #     U = np.random.randn(U.shape[0], U.shape[1])
    #     b_tag = np.random.randn(b_tag.shape[0])
    #     gradient_check(_loss_and_b_grad, b)
    #     gradient_check(_loss_and_W_grad, W)
    #     gradient_check(_loss_and_U_grad, U)
    #     gradient_check(_loss_and_b_tag_grad, b_tag)
    #     gradient_check(_loss_and_V_grad, V)
    #     gradient_check(_loss_and_c_grad, c)

    train_data = utils.read_data(r'train')
    dev_data = utils.read_data(r'dev')
    test_data = utils.read_data(r'test')
    in_dim = len(utils.vocab)
    out_dim = len(utils.L2I)
    dimensions = [in_dim, 30, 40, 20, out_dim]

    params = create_classifier(dimensions)

    num_iterations = 100
    learning_rate = 0.9

    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    #get_tag(test_data, trained_params)

