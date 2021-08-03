import numpy as np


class LinearGaussianModel:
    """A pre-trained Linear Gaussian Model."""

    def __init__(self, weights_path):
        """
        Args:
            weights_path: path to pre-specified choice of W
        """
        # Pre-load an existing model
        self.W = np.loadtxt(weights_path, delimiter=",")
        self.b = np.loadtxt("b.txt", delimiter=",").reshape(-1, 1)
        (self.d, self.m) = self.W.shape
        self.gamma = 0.2

    def log_likelihood(self, x):
        """Compute the log-likelihood of x.

        Args:
            x: An observed sample. Shape (d, 1)

        Returns:
            The log-likelihood, log p(x). Scalar value.
        """
        # *** START CODE HERE ***

        # Generating identity matrix of size d
        I_d = np.identity(self.d)

        # Obtain mu and sigma. Using results from the assignment
        # Reshaping mu so it's a (dim,) vector instead of (dim,1) to make compatible with compute_normal
        mu = self.b.flatten()
        sigma = self.W.dot(self.W.T) + self.gamma**2 * I_d

        # Compute log likelihood
        log_likelihood = np.log(compute_normal(x.T, mu, sigma))

        return log_likelihood[0]
        # *** END CODE HERE ***

    def elbo(self, x, qm, qr, num_samples):
        """Compute the Evidence Lower Bound of x using the distribution q.

        Args:
            x: An observed sample. Shape (d, 1).
            qm: The mean of the factorized Gaussian distribution q(z). Shape (m, 1).
            qr: The log(stddev) of the factorized Gaussian distribution q(z). Shape (m, 1).
            num_samples: number of samples used for monte carlo estimation.

        Returns:
            The evidence lower bound of x. Scalar value.
        """
        # *** START CODE HERE ***

        ##################################################
        # Exact calculation of KL divergence
        ##################################################
        KL = KL_divergence(qm, qr)

        ##################################################
        # Monte Carlo estimate of reconstruction
        ##################################################

        # Reconstructing the variance of z first
        var_qr = np.exp(qr) * np.identity(self.m)
        # Generating all z_i, this is a (num_samples, m) matrix where m is # of latent dimensions
        z_sample = np.random.multivariate_normal(qm.flatten(), var_qr, num_samples)
        # Now we can calculate the mean of x, given by (Wz+b)
        fz = np.zeros((num_samples, self.d))

        # Lets calculate the variance of the normal distribution for x, which we'll use later
        var_x = self.gamma**2 * np.identity(self.d)

        # Initializing matrix where we'll store probabilities for every sample
        prob_matrix = np.zeros(num_samples)

        # Lets generate the mean for the distribution x|z
        for i in range(num_samples):
            fz[i] = self.W.dot(z_sample[i]) + self.b.flatten()
            # Now we have a (num_samples, d) matrix where we have the mean of each d dimension for each sample
            # x|z is distributed as N(f(z), Y^2 I_d)
            prob_x = compute_normal(x.T, fz[i], var_x)
            prob_matrix[i] = np.log(prob_x)

        MCE = (1/num_samples) * np.sum(prob_matrix)

        ##################################################
        # Putting it all together: calculating evidence lower bound
        ##################################################
        elbo = MCE - KL

        return elbo
        # *** END CODE HERE ***

    def sgvb(self, x, qm_init, qr_init, num_samples, num_iter=1000, lr=0.001):
        """Perform stochastic gradient variational bayes.

        Args:
            x: An observed sample. Shape (d, 1).
            qm_init: Initial guess for the mean of the factorized Gaussian distribution q(z). Shape (m, 1).
            qr_init: Initial guess for the log(stddev) of the factorized Gaussian distribution q(z). Shape (m, 1).
            num_samples: Number of samples used for monte carlo estimation.
            num_iter: Number of gradient ascent steps.
            lr: Learning rate.

        Returns:
            qm: The optimized mean after gradient ascent. Shape (m, 1).
            qr: The optimized log(stddev) after gradient ascent. Shape (m, 1).
        """
        # *** START CODE HERE ***

        qm = qm_init
        qr = qr_init

        for i in range(num_iter):
            # print("-"*30)
            # print(f"This is iteration {i}")
            ##################################################
            # Gradient of KL divergence w.r.t. mu and sigma
            ##################################################
            # In the assignment, we computed the closed-form solution for the gradient of KL divergence
            grad_kl_mu = qm
            grad_kl_sigma = np.exp(qr) - (1/np.exp(qr))
            grad_kl_rho = grad_kl_sigma * np.exp(qr)

            ##################################################
            # Gradient of Monte Carlo
            ##################################################
            # We can use the simplified version of the gradient with respect to z
            # First, lets generate the N samples that we'll use for monte carlo estimation
            # Second, we can calculate the probability of x under those samples
            # Lastly, we can sum over all samples to compute the expectation over all samples

            # Generating samples for monte carlo, this is a (num_samples, m) matrix where m is # of latent dimensions
            var_qr = np.exp(qr)**2 * np.identity(self.m)
            # var_qr2 = (np.exp(qr) ** 2).flatten()

            e_mean = np.zeros(self.m)

            # e_sample = np.zeros((num_samples, self.m))
            # for m in range(num_samples):
            #     e_sample[m] = np.random.normal(e_mean, var_qr2)
            e_sample = np.random.multivariate_normal(e_mean, var_qr, num_samples)

            z = qm + (np.exp(qr)) * e_sample.T
            z = z.T

            scalar = (1 / self.gamma**2)
            step1 = (self.W.T).dot(x)
            step2 = (self.W.T).dot(self.W)
            step3 = (self.W.T).dot(self.b)

            mu_compute = np.zeros((num_samples,self.m))
            sigma_compute = np.zeros((num_samples, self.m))

            for m in range(num_samples):
                new_step2 = step2.dot(z[m]).reshape((self.m, 1))
                mu_compute[m] = scalar * (step1 - new_step2 - step3).flatten()
                sigma_compute[m] = mu_compute[m] * e_sample[m]

            # Taking average across all examples to compute mu and sigma
            grad_mc_mu = (1/num_samples) * np.sum(mu_compute, axis=0).reshape((self.m,1))
            rho_compute = sigma_compute * np.exp(qr).flatten()
            grad_mc_rho = (1/num_samples) * np.sum(rho_compute, axis=0).reshape((self.m, 1))

            ##################################################
            # Putting it all together: Gradient of ELBO
            ##################################################
            grad_qm = grad_mc_mu - grad_kl_mu
            grad_qr = grad_mc_rho - grad_kl_rho

            ##################################################
            # Performing gradient ascent
            ##################################################
            qm += lr * grad_qm
            qr += lr * grad_qr

        return qm, qr
        # *** END CODE HERE ***


# Optional: place helper functions here
# *** START CODE HERE ***
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

def KL_divergence(qm, qr):
    """ Calculates the KL divergence D(q||p_z)

    Arguments:
        qm: The mean of the factorized Gaussian distribution q(z). Shape (m, 1).
        qr: The log(stddev) of the factorized Gaussian distribution q(z). Shape (m, 1)

    Returns:
        KL: divergence between both distributions. Scalar.
    """
    log_sigma = - qr
    inner = (1 / 2) * (np.exp(qr) ** 2 + qm ** 2 - 1)
    KL = np.sum(log_sigma + inner)

    return KL

# *** END CODE HERE ***


def experiment(xid):
    print("*" * 10 + "BEGIN EXPERIMENT" + "*" * 10)
    print("Experiment {0}: Model-{0} applied to observation x{0}".format(xid))
    print("Model-{0} loaded with W from W{0}.txt".format(xid))
    lgm = LinearGaussianModel(weights_path="W{}.txt".format(xid))
    x = np.loadtxt("x{}.txt".format(xid), delimiter=",").reshape(-1, 1)
    print("Model-{0} ln p(x{0}):".format(xid), lgm.log_likelihood(x))

    qm_init = np.zeros((lgm.m, 1))
    qr_init = np.zeros((lgm.m, 1))
    print("Model-{0} ELBO(x{0}) using initial q:".format(xid), lgm.elbo(x, qm_init, qr_init, num_samples=10000))

    qm, qr = lgm.sgvb(x, qm_init, qr_init, num_samples=10, num_iter=1000, lr=0.001)
    print("Model-{0} ELBO(x{0}) using optimized q:".format(xid), lgm.elbo(x, qm, qr, num_samples=10000))
    print("*" * 10 + "END EXPERIMENT" + "*" * 10 + "\n")



if __name__ == "__main__":
    experiment(1)
    experiment(2)
