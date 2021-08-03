import numpy as np

A = np.array([[1, 0], [0, 2]])
theta_0 = np.array([-1, 0.5])

def J(theta):
    return theta.T.dot(A).dot(theta)

def update_theta(theta, lr):
    """Problem: given the current value of theta and the learning rate lr,
    you should return the new value of theta obtained by running 1 iteration
    of the gradient descend algorithm.

    Args:
        theta: the current theta
        lr: the learning rate

    Returns:
        the new value of theta after 1 iteration of gradient descend
    """
    # *** START CODE HERE ***

    # Lets first calculate the derivative of the loss function
    gradient = 2 * np.dot(A,theta)

    # Now we can calculate the gradient descent rule
    theta_new = theta - lr * gradient

    return theta_new
    # *** END CODE HERE ***

def gradient_descend(J, theta_0, lr, update_theta, epsilon=1e-50):
    """Write the gradient descend algorithm using the parameters.
    You can stop the algorithm when either:
        1. the absolute difference of J(theta^[t]) and J(theta^[t-1]) is less than epsilon or
        2. the loss function J(theta^[t]) is bigger than 1e20

    Args:
        J: the objective function
        theta_0: the initial theta
        lr: the learning rate
        update_theta: the theta update function, which you implemented above
        epsilon: we stop when the absolute loss function differences is below this value
    """
    theta = theta_0
    # *** START CODE HERE ***

    # Initialize loss and loss difference
    loss = J(theta)
    diff = 1
    counter = 0

    # Initiate loop that stops when conditions are met
    while diff >= epsilon and loss < 1e20:
        # Run iteration of update theta
        theta = update_theta(theta, lr)

        # Computing objective function
        loss_new = J(theta)

        # Computing difference between loss functions
        diff = np.abs(loss_new - loss)

        # Updating loss
        loss = loss_new

        # Printing debug statements
        print(f"Iteration: {counter}")
        print(f"Loss: {loss}")
        print(f"Difference in loss: {diff}")
        print("---------------")

        # Updating iteration counter
        counter += 1

    # *** END CODE HERE ***
    return theta

if __name__ == "__main__":
    theta = gradient_descend(J, theta_0, 1e-2, update_theta)
    assert np.isclose(theta[0], theta[1]), f"elements of theta {theta} is not close"
    assert all(abs(theta_i) < 1e-24 for theta_i in theta), f"elements of theta {theta} is too far from the optimal value"
    print("All sanity checks passed")

