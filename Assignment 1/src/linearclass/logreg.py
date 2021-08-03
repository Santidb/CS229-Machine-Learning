import numpy as np
import matplotlib.pyplot as plt
import util

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path

    # Loading validation set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    # Training logistic regression model on training sample
    reg = LogisticRegression()
    reg.fit(x_train, y_train)

    # Generating predictions on validation sample
    y_val_hat = reg.predict(x_val)

    # Renaming filepath from .txt to .png and generating plot
    save_path_img = save_path[:-3]+'png'
    util.plot(x_val, y_val, reg.theta, save_path_img)

    # Saving predictions on eval set
    np.savetxt(save_path, y_val_hat)

    # *** END CODE HERE ***

# Defining sigmoid function to simplify math
def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=False):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Capture number of observations
        num_obs = x.shape[0]

        # If theta_0 is None, then initialize with the vector of 0
        if self.theta == None:
            self.theta = np.zeros((x.shape[1]))

        # Initialize the norm at 1 and iteration counter
        norm = 1
        k = 1

        # Begin loop
        while norm > self.eps and k <= self.max_iter:

            # Calculating the sigmoid of this iteration
            sig = sigmoid(np.dot(self.theta, x.T))

            # Now we want to calculate the gradient of the logistic function
            # Use broadcasting to multiply by x, then sum all observations and divide by number of observations
            gradient = np.sum((sig - y) * x.T, axis=1) / num_obs

            # We also want to compute the Hessian
            hessian = np.dot(x.T * sig * (1 - sig), x) / num_obs

            # Computing the inverse of the hessian
            inv_hessian = np.linalg.inv(hessian)

            # Updating theta
            new_theta = self.theta - self.step_size * np.dot(inv_hessian, gradient)

            # Calculating norm
            norm = np.sum(np.abs(new_theta - self.theta))

            # Updating theta
            self.theta = new_theta

            # Debugging prints if verbose is True
            if self.verbose is True:
                print(f"Norm is {norm} at iteration {k}")

            # Updating iteration
            k += 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Computing logistic regression
        y_hat = sigmoid(np.dot(self.theta, x.T))

        return y_hat
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
