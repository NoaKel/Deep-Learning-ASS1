import numpy as np

STUDENT={'name': 'Noa Yehezkel Lubin',
         'ID': '305097552'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    res = np.exp(x - np.max(x))/np.sum(np.exp(x - np.max(x))); # softmax(x+c)=softmax(x)
    return res
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    probs = softmax(np.dot(x,W)+b);
    
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params, lambda_regul=0):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    y_gag = classifier_output(x,params)
    loss = -np.log(y_gag[y]) + lambda_regul * (0.5 * np.sum(W**2) + 0.5 * np.sum(b**2))
    #   grad W,b calculation
    gb = y_gag
    gb[y] = -(1 - y_gag[y])
    gW = np.outer(x,gb) + lambda_regul * W
    gb+= lambda_regul * b
    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    test4 = softmax(1)
    print test4
    assert np.amax(np.fabs(test4 - 1)) <= 1e-6

    test5 = softmax(np.array([0,0]))
    print test5
    assert np.amax(np.fabs(test5 - np.array([0.5, 0.5]))) <= 1e-6
    
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
