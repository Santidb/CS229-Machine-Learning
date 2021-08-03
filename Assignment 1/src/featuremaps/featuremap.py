import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Solving using normal equations
        # np.linalg.solve(a,b) solves the equation Ax=b, or x = A^-1 * b
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***

        # Lets first generate the vector with k's
        k_vec = np.arange(k+1)

        # Obtain the column vector from x
        x_vec = X[:,[1]]

        # We want to map attributes x to a feature map x_hat
        polymap = x_vec**k_vec

        return polymap
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***

        # We want to create a polymap with a sin(x) term at the end

        # Lets first generate the vector with k's
        k_vec = np.arange(k + 1)

        # Obtain the column vector from x
        x_vec = X[:, [1]]

        # We want to map attributes x to a feature map x_hat
        polymap = x_vec ** k_vec

        # Computing the sin term
        sine = np.sin(x_vec)

        # Adding everything back into the vector
        sinemap = np.append(polymap, sine, axis=1)

        return sinemap
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Obtaining the prediction
        y_hat = np.dot(X, self.theta.T)

        return y_hat
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***

        # Training linear regression
        model = LinearModel()

        # Creating feature map with sine or polynomial of order k
        if sine is True:
            feature_x = model.create_sin(k, train_x)
        else:
            feature_x = model.create_poly(k, train_x)

        # Fitting model with polynomial feature map
        model.fit(feature_x, train_y)

        # Creating feature map for plot_x
        if sine is True:
            feature_plot_x = model.create_sin(k, plot_x)
        else:
            feature_plot_x = model.create_poly(k, plot_x)

        # Predicting values
        plot_y = model.predict(feature_plot_x)

        # *** END CODE HERE ***

        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression')

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***

    # Running experiment with k=3
    run_exp(train_path, ks=[3], filename='featuremap_4b.png')

    # Running experiment with k = [3, 5, 10, 20]
    run_exp(train_path, ks=[3,5,10,20], filename = 'featuremap_4c.png')

    # Running experiment with k = [0,1,2,3,5,10,20] and sine=true
    run_exp(train_path, sine=True, ks=[0,1,2,3,5,10,20], filename='featuremap_4d.png')

    # Running experiment with small data
    run_exp(small_path, ks=[1,2,5,10,20], filename='featuremap_4e.png')

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
