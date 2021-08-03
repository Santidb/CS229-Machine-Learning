import numpy as np
import matplotlib.pyplot as plt
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path

    # Load validation dataset. We want to add intercept to this to compute predictions easily.
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    # Experimenting with transformations for x(i)
    # x_train_log = np.log(x_train + 3.5)
    # x_val_log = np.c_[ x_val[:, 0], np.log(x_val[:, 1:] + 3.5) ]

    # Initialize the model and train the parameters
    model = GDA()
    model.fit(x_train, y_train)

    # Now we can generate predictions
    y_val_hat = model.predict(x_val)

    # Renaming filepath from .txt to .png and generating plot
    save_path_img = save_path[:-3]+'png'
    util.plot(x_val, y_val, model.theta, save_path_img)

    # Saving predictions on eval set
    np.savetxt(save_path, y_val_hat)

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters

        # Number of observations
        obs = x.shape[0]
        dim = x.shape[1]

        y = np.reshape(y, (obs,1))

        # Obtaining parameters
        phi = (1/obs)*np.sum(y)
        mu_0 = np.sum( x * (y==0), axis=0) / np.sum(y==0)
        mu_1 = np.sum( x * (y==1), axis=0) / np.sum(y==1)

        # Lets combine both mu's in a matrix that selects mu_1 when y==1 and mu_0 when y==0
        mu_comb = mu_0.reshape(1, dim) * (y == 0) + mu_1.reshape(1, dim) * (y == 1)

        # Lets calculate sigma on a loop over all iterations
        sigma = 0
        for i in range(obs):
            sigma += np.dot((x[i] - mu_comb[i]).reshape(dim,1), (x[i] - mu_comb[i]).reshape(1,dim))
        sigma *= (1/obs)

        # Now we can calculate theta_0 and theta
        sigma_inv = np.linalg.inv(sigma)
        theta_T = np.dot((mu_1 - mu_0), sigma_inv)

        # Lets disaggregate the steps to calculate theta_0
        step_1 = np.dot((mu_0 + mu_1).reshape((1,dim)), sigma_inv)
        step_2 = np.dot(step_1, (mu_0 - mu_1).reshape(dim,1))[0]
        log_phi = np.log((1-phi)/phi)
        theta_0 = (1/2) * step_2 - log_phi

        # Aggregating theta_0 and theta_T in the same vector
        self.theta = np.concatenate((theta_0, theta_T))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Dimensions
        obs = x.shape[0]

        # Computing predictions using sigmoid function
        y_hat = 1 / (1 + np.exp(-np.dot(self.theta, x.T)))

        return y_hat

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
