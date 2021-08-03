import numpy as np
import util

# Noise ~ N(0, sigma^2)
sigma = 0.5
# Dimension of x
d = 500
# Theta ~ N(0, eta^2*I)
eta = 1/np.sqrt(d)
# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

def analytical_solver(x_train, y_train, scale):
    """ Compute the analytical solution to find the parameters that minimize the loss function.

    Args:
        x_train: training examples
        y_train: training labels
        scale: lambda for regularization

    Returns: theta optimal
    """
    x_sq = x_train.T.dot(x_train)

    scalar = 2 * sigma**2 * scale
    step2 = scalar * np.identity(x_sq.shape[0])

    step3 = x_sq + step2
    step4 = x_train.T.dot(y_train)

    # Using different ways of inverting matrices due to numerical instability
    if scale == 0:
        inv_mat = np.linalg.pinv(step3)
        theta = inv_mat.dot(step4)
    else:
        theta = np.linalg.solve(step3, step4)

    return theta

def MSE(x_val, y_val, theta):
    """ Calculate MSE on validation dataset

    Args:
        x_val: validation examples
        y_val: validation labels
        theta: optimal theta from trained model

    Returns: Minimum Squared Error
    """
    # Size of dataset
    n_val = x_val.shape[0]

    # norm of predictions
    norm = np.linalg.norm(x_val.dot(theta) - y_val)

    # calculating mse
    mse = (1 / n_val) * norm**2

    return mse

def ridge_regression(train_path, validation_path):
    """Problem 5 (d): Parsimonious double descent.
    For a specific training set, obtain theta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in scale_list.
    """
    # *** START CODE HERE ***

    # Load datasets
    x_train, y_train = util.load_dataset(train_path)
    x_val, y_val = util.load_dataset(validation_path)

    # Looping over all lambdas
    scale_opt = 1 / (2*(eta**2))
    scale_new_list = np.asarray(scale_list) * scale_opt

    # Initializing list with validation MSEs
    val_err = []

    # Initialize loop over lambdas (scale)
    for scale in scale_new_list:

        # Find optimal theta using closed form solution on training dataset
        theta = analytical_solver(x_train, y_train, scale)

        # Evaluate MSE on validation dataset
        error = MSE(x_val, y_val, theta)

        # Store validation MSE on validation error list
        val_err.append(error)

    # *** END CODE HERE
    return val_err

if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
        print(n)
    val_err = np.asarray(val_err).T
    util.plot(val_err, 'doubledescent.png', n_list)
