import numpy as np

STUDENT={'name': 'Noa Yehezkel Lubin',
         'ID': '305097552'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classifier_output(x, params):
    W, U, b, b_tag = params
    probs = softmax(np.dot(np.tanh(np.dot(x,W) + b),U) + b_tag)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params,lambda_regul=0):
    W, U, b, b_tag = params

    y_gag = softmax(np.dot(np.tanh(np.dot(x,W) + b),U) + b_tag)
    loss = -np.log(y_gag[y]) + lambda_regul * (0.5 * np.sum(W**2) + 0.5 * np.sum(b**2) + 0.5 * np.sum(b_tag ** 2)+ 0.5 * np.sum(U ** 2))
    gb_tag = y_gag
    gb_tag[y] = -(1 - y_gag[y])
    gU = np.outer(np.tanh(np.dot(x,W) + b), gb_tag)
    gb = np.dot(gb_tag,U.T) * (np.array([1]*b.shape[0]) - np.tanh(np.dot(x,W) + b) ** 2)
    gW = np.outer(x,gb)

    #   add regularization to gradient
    gb += lambda_regul * b
    gb_tag += lambda_regul * b_tag
    gW += lambda_regul * W
    gU += lambda_regul * U

    return loss, [gW,gU,gb,gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    eps = -6 ** 0.5
    W = np.random.uniform(-eps / (in_dim+hid_dim) ** 0.5, eps / (in_dim+hid_dim) ** 0.5, (in_dim, hid_dim))
    U = np.random.uniform(-eps / (hid_dim+out_dim) ** 0.5, eps / (hid_dim+out_dim) ** 0.5, (hid_dim, out_dim))
    b = np.random.uniform(-eps / hid_dim ** 0.5, eps / hid_dim ** 0.5,  hid_dim)
    b_tag = np.random.uniform(-eps / out_dim ** 0.5, eps / out_dim ** 0.5,out_dim)
    params = [W,U,b,b_tag]
    return params


if __name__ == '__main__':
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

    W, U, b, b_tag = create_classifier(3, 4, 5)


    def _loss_and_W_grad(W):
        global b, b_tag, U
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, U, b, b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W, b_tag,U
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, U, b, b_tag])
        return loss, grads[2]

    def _loss_and_U_grad(U):
        global W,b,b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, U, b, b_tag])
        return loss, grads[1]

    def _loss_and_b_tag_grad(b_tag):
        global W,U,b
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, U, b, b_tag])
        return loss, grads[3]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)



