import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path

    # Loading validation dataset
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    # Setting up the model and calculating optimal parameters through gradient ascent
    model = PoissonRegression(step_size = lr)
    model.fit(x_train, y_train)

    # Predicting values on validation dataset
    y_val_hat = model.predict(x_val)

    # Creating scatter plot
    plot_poisson(y_val, y_val_hat, 'poisson_pred.png')


    # Saving results to save_path
    np.savetxt(save_path, y_val_hat)

    # *** END CODE HERE ***

def plot_poisson(y_true, y_predicted, save_path):
    """ Plot true and predicted values from a Poisson regression

    Args:
        y_true: True counts, will be discrete
        y_predicted: Predicted counts, will be continuous
        save_path: Path to save plot image
    """
    # Plot dataset
    plt.scatter(y_true, y_predicted, alpha=0.5, c='green')

    # Adding formatting
    plt.title('True vs. Predicted counts from Poisson Regression', fontsize=14)
    plt.xlabel('True count', fontsize=12)
    plt.ylabel('Predicted count', fontsize=12)
    plt.savefig(save_path)

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Obtaining dimensions
        obs = x.shape[0]
        dim = x.shape[1]

        # If theta_0 is None, then initialize with the vector of 0
        if self.theta == None:
            self.theta = np.zeros((dim))


        # Lets initialize a loop until we hit the max number of iterations or converge
        norm = 1
        iter = 0

        while norm > self.eps and iter <= self.max_iter:

            # Calculating the batch gradient
            exp = np.exp(np.dot(x, self.theta))

            grad = (y - exp).reshape(obs, 1) * x
            sum_grad = grad.sum(axis=0)

            # Updating theta
            new_theta = self.theta + self.step_size * sum_grad

            # Updating norm
            norm = np.sum(np.abs(new_theta - self.theta))

            # Storing new theta
            self.theta = new_theta

            # Updating iteration
            iter += 1

            # Printing debug string if verbose is True
            if self.verbose is True:
                print(f"Norm is {norm} on iteration {iter}")

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Generating predictions
        y_hat = np.exp(np.dot(self.theta, x.T))

        return y_hat
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
