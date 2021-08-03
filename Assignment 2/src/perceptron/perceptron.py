import math

import matplotlib.pyplot as plt
import numpy as np

import util


def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron. The state of the perceptron can include
    anything the model must store to make predictions.
    """

    # *** START CODE HERE ***
    beta = None
    kernel_dict = {}
    x_matrix = None

    dict = {
        "beta": beta,
        "kernel": kernel_dict,
        "x_matrix": x_matrix
    }

    return dict
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Peform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance

    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    # Unpacking values from previous state
    beta = state["beta"]
    kernel_dict = state["kernel"]
    x_matrix = state["x_matrix"]

    # Computing kernels for this iteration
    num_iter = beta.shape[0]
    k_vector = [0] * (num_iter)
    # Populating kernel vector: K(x_1, x_i+1), K(x_2, x_i+1), ..., K(x_i, x_i+1)
    for i in range(num_iter):
        k_vector[i] = kernel(x_matrix[i], x_i)

    # Computing prediction
    input = beta * k_vector
    y_hat = sign(np.sum(input))

    return y_hat
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    # Unpacking values from previous state
    beta = state["beta"]
    kernel_dict = state["kernel"]
    x_matrix = state["x_matrix"]

    # Adding x_i to x_matrix
    if x_matrix is None:
        x_matrix = np.zeros((1, x_i.shape[0]))
        x_matrix[0] = x_i
    else:
        x_matrix = np.append(x_matrix, [x_i], axis=0)

    # Iterations so far (starting from 1, not from 0)
    num_iter = x_matrix.shape[0]

    # Computing kernels
    # First, initializing vector where we will store kernels for this iteration
    k_vector = [0]*(num_iter)
    # Populating kernel vector: K(x_1, x_n+1), K(x_2, x_n+1), ..., K(x_n+1, x_n+1)
    for i in range(num_iter):
        k_vector[i] = kernel(x_matrix[i], x_i)

    # Storing kernels of this iteration in the kernel dictionary
    kernel_dict[num_iter] = k_vector

    # Updating value of beta
    if beta is None:
        input = 0
        beta_i = learning_rate * ( y_i - sign(input))
        beta = np.array([beta_i])
    else:
        input = np.sum(beta * k_vector[:-1])
        beta_i = learning_rate * ( y_i - sign(input))
        beta = np.append(beta, beta_i)

    # Updating state
    state["x_matrix"] = x_matrix
    state["beta"] = beta
    state["kernel"] = kernel_dict

    # *** END CODE HERE ***


def sign(a):
    """Gets the sign of a scalar input."""
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)

def non_psd_kernel(a, b):
    """An implementation of a non-psd kernel.

    Args:
        a: A vector
        b: A vector
    """
    if(np.allclose(a,b,rtol=1e-5)):
        return -1
    return 0

def train_perceptron(kernel_name, kernel, learning_rate):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/perceptron/perceptron_{kernel_name}_predictions.txt.
    The output plots are saved to src/perceptron/perceptron_{kernel_name}_output.pdf.

    Args:
        kernel_name: The name of the kernel.
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    train_x, train_y = util.load_csv('train.csv')

    state = initial_state()

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv('test.csv')

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.savefig('perceptron_{}_output.png'.format(kernel_name))

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('perceptron_{}_predictions'.format(kernel_name), predict_y)


def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)
    train_perceptron('non_psd', non_psd_kernel, 0.5)
    plt.show()



if __name__ == "__main__":
    main()
