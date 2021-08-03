import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the unlabeled data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    # Obtain dimensions of unlabeled data
    n = x.shape[0]
    dim = x.shape[1]

    # Put each data point into a sample based on uniform distribution
    ind_sample = np.random.randint(0, K, n)

    # Create samples, of different sizes, based on ind_sample
    samples = []
    for k in np.arange(K):
        sample_k = x[ind_sample==k]
        samples.append(sample_k)

    samples = np.asarray(samples)

    # Obtain sample mean and covariance for each group
    # Mean is a Kxd matrix (4x2x1) while covariance is a (4 x dim x dim) matrix (4 x 2 x 2)
    mu = np.ones((K, dim))
    sigma = np.ones((K, dim, dim))

    for k in np.arange(K):
        mu[k] = samples[k].mean(axis=0)
        sigma[k] = np.cov(samples[k].T)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, (1/K))

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full([n, K], (1 / K))

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    # Obtain dimensions of unlabeled data
    n = x.shape[0]
    dim = x.shape[1]

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE

        # Update prev_ll
        prev_ll = ll

        # (1) E-step: Update your estimates in w

        # We will store the numerator of each cluster j in the loop
        w_num = np.zeros((n, K))
        for k in np.arange(K):
            det_sigma = np.linalg.det(sigma[k])
            inv_sigma = np.linalg.inv(sigma[k])
            scalar = 1 / (det_sigma**(1/2))
            norm = x - mu[k]
            inside = norm.dot(inv_sigma).dot(norm.T)
            numerator = scalar * np.exp(-(1/2)*inside) * phi[k]
            # Store the numerator in w_num
            w_num[:, k] = np.diag(numerator)

        # calculate the denominator by summing over all clusters
        w_denominator = np.sum(w_num, axis=1)

        # Compute w
        w = w_num / w_denominator.reshape((n, 1))

        # (2) M-step: Update the model parameters phi, mu, and sigma

        for k in np.arange(K):

            # mu
            mu_num = np.sum(w[:, k].reshape((n,1)) * x, axis=0)
            mu_denom = np.sum(w[:,k])
            mu[k] = mu_num / mu_denom

            # phi
            phi[k] = (1/n) * np.sum(w[:,k])

            # sigma
            # Using updated "new" mu
            norm = x - mu[k]
            # Computing weights x demeaned matrix
            sig_mat = w[:,k].reshape((n,1)) * norm
            # Computing numerator. We do not need to explicitly sum because it is handled by the matrix operation
            sig_num = sig_mat.T.dot(norm)
            sigma[k] = sig_num / np.sum(w[:,k])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        ll = 0
        for k in np.arange(K):
            log_num = compute_normal(x, mu[k], sigma[k]) * phi[k]
            log_term = log_num / w[:,k]
            likelihood_k = w[:,k] * np.log(log_term)
            ll += np.sum(likelihood_k)

        # Updating iteration
        it += 1

        # debug print
        print(f"Iteration {it}: log-likelihood {ll}")

        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples_unobs, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples_unobs, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Obtain dimensions of unlabeled data
    n = x.shape[0]
    dim = x.shape[1]
    n_tilde = x_tilde.shape[0]

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***

        # Update prev_ll
        prev_ll = ll

        # (1) E-step: Update your estimates in w

        # We will store the numerator of each cluster j in the loop
        w_num = np.zeros((n, K))
        for k in np.arange(K):
            det_sigma = np.linalg.det(sigma[k])
            inv_sigma = np.linalg.inv(sigma[k])
            scalar = 1 / (det_sigma ** (1 / 2))
            norm = x - mu[k]
            inside = norm.dot(inv_sigma).dot(norm.T)
            numerator = scalar * np.exp(-(1 / 2) * inside) * phi[k]
            # Store the numerator in w_num
            w_num[:, k] = np.diag(numerator)

        # calculate the denominator by summing over all clusters
        w_denominator = np.sum(w_num, axis=1)

        # Compute w
        w = w_num / w_denominator.reshape((n, 1))

        # (2) M-step: Update the model parameters phi, mu, and sigma

        for k in np.arange(K):

            # indicator function
            indicator = (z_tilde==k)

            # mu
            mu_unsup_num = np.sum(w[:, k].reshape((n,1)) * x, axis=0)
            mu_tilde_num = np.sum(indicator * x_tilde, axis=0)
            mu_num = mu_unsup_num + alpha * mu_tilde_num

            mu_unsup_denom = np.sum(w[:,k])
            mu_tilde_denom = np.sum(indicator)
            mu_denom = mu_unsup_denom + alpha * mu_tilde_denom

            mu[k] = mu_num / mu_denom

            # phi
            phi[k] = (1/(n+alpha*n_tilde)) * (np.sum(w[:,k]) + alpha * np.sum(indicator))

            # sigma
            # Unsupervised numerator
            norm_unsup = x - mu[k]
            sig_mat_unsup = w[:,k].reshape((n,1)) * norm_unsup
            sig_num_unsup = sig_mat_unsup.T.dot(norm_unsup)

            # Supervised numerator
            norm_sup = x_tilde - mu[k]
            sig_mat_sup = indicator * norm_sup
            sig_num_sup = sig_mat_sup.T.dot(norm_sup)

            # Sigma numerator
            sig_num = sig_num_unsup + alpha * sig_num_sup

            # Unsupervised denominator
            sig_denom_unsup = np.sum(w[:,k])
            sig_denom_sup = np.sum(indicator)
            sig_denom = sig_denom_unsup + alpha * sig_denom_sup

            sigma[k] = sig_num / sig_denom


        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        ll = 0
        for k in np.arange(K):
            # Unsupervised part
            log_num = compute_normal(x, mu[k], sigma[k]) * phi[k]
            likelihood_k = w[:, k] * (np.log(np.clip(log_num, 1e-12, None)) - np.log(np.clip(w[:,k], 1e-12, None)))
            ll += np.sum(likelihood_k)

            # Supervised part
            indicator = (z_tilde==k)

            log_term_sup = np.log(compute_normal(x_tilde, mu[k], sigma[k]) * phi[k]).reshape((n_tilde, 1))
            # log_term_sup = np.log(compute_normal(x_tilde, mu[k], sigma[k])).reshape((n_tilde, 1))
            ll += alpha * np.sum(log_term_sup * indicator)

        # Updating iteration
        it += 1

        # debug print
        print(f"Iteration {it}: log-likelihood {ll}")

        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z

def compute_normal(x, mu, sigma):
    """ Compute the calculations for the normal distribution

    Args:
        x: original data shape (n, dim)
        mu: mean of data shape (dim, )
        sigma: covariance of data shape (dim, dim)

    Returns:
        normal: NumPy array shape (n, 1)
    """
    # Obtaining dimensions of data
    n = x.shape[0]
    dim = x.shape[1]

    pi_term = (2*np.pi)**(dim/2)
    sig_det = np.linalg.det(sigma)**(1/2)
    scalar = 1 / (pi_term * sig_det)

    sigma_inv = np.linalg.inv(sigma)
    norm = (x - mu)
    exp_term = norm.dot(sigma_inv).dot(norm.T)

    matrix = scalar * np.exp( (-1/2) * exp_term)
    normal = np.diag(matrix)

    return normal

if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
