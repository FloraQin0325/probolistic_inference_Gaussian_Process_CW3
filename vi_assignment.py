# import inline as inline
# import matplotlib

import numpy as np


def generate_data(N=1000, D=2, K=None, mu=None, sigma=1., dtype=np.float64):
    """
    Generates samples from a Mixture of Gaussian (MoG).
    Do not change this function!
    Parameters
    ----------
    N: int
       number of samples
    K: int
       number of components
    mu: vector of shape (K) or array_like of shape (K x D)
        mean of components
    sigma: float
           standard deviation of each Gaussian component

    Returns
    -------
    samples: ndarray of shape(N x D)
             Samples from MoG distribution.
    c: 1D ndarray of shape (N,)
       Cluster assignment corresponding to samples
    """
    if mu is None:
        assert K is not None, "Must pass in either mu or K"
        mu = np.random.normal(np.zeros((K, D)), np.ones((K, D)) * 2.)
    else:
        K = mu.shape[0]

    mu = mu.reshape((1, K, -1)).repeat(N, axis=0)
    c = np.random.choice(K, N)
    mu_c = mu[np.arange(N), c, :]
    samples = np.random.normal(mu_c, np.ones_like(mu_c) * sigma)
    return samples, c


class GMM_CAVI(object):
    """
    Implements the Gaussian Mixture Model (GMM) with
    Coordinate Ascent Variational Inference. The means of mixture
    components are assumed to have a Gaussian prior N(0, pmu_var),
    the cluster indicators are assumed to have a uniform prior and
    the likelihood variance term is assumed to be known and fixed
    as the identity matrix.
    The variational distribution is specified by a gaussian
    for the means of the components and a categorical distribution
    over the mixture components:
    ----------
    m: mean of gaussian of the mixture components q(mu) - K x D
    s2: variance of gaussian of the mixture components q(mu) - K x D
    pi: probabilities for the categorical over mixture components q(c) - N x K
    ----------
    Initialisation Parameters
    ----------
    X: float,
       D dimensional dataset
    K: int,
       number of mixture components
    pmu_var: float,
       prior variance on mean of components p(mu)
    ----------
    """

    def __init__(self, X, K, pmu_var=1., seed=31415):
        """
        Initialization of (variational) parameters.
        Do not change this function!
        """
        self.X = X
        self.K = K
        self.N, self.D = X.shape

        self.rng = np.random.RandomState(seed)

        self.pmu_var = np.float64(self.D * [pmu_var])

        self.m = np.zeros(
            (self.K, self.D)) + self.rng.randn(self.K, self.D) * 1e-2
        self.s2 = np.ones_like(self.m)

        self.pi = self.rng.dirichlet(
            self.rng.randint(1, 10, size=self.K), self.N)

    def update_m(self):
        """
        Returns the optimal factor for the mean of
        the variational distibution over the cluster means q*(mu_k)
        - Ignoring constant terms!
        -------
        m: K x D
        """
        m = 0
        #######################################################################
        # TODO: Implement the update for the mean of q*(mu_k) here using      #
        #       the appropriate class variables.                              #
        #######################################################################

        m = np.zeros((self.K, self.D))

        for i in range(self.K):
            sum_pi_x = 0
            sum_pi = 0
            for j in range(self.N):
                sum_pi_x += self.pi[j, i]*self.X[j, :].reshape(1, self.D)
                sum_pi += self.pi[j,i]
            m[i,:] = sum_pi_x/(sum_pi + 1/self.pmu_var[0])
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return m

    def update_s2(self):
        """
        Returns the optimal factor for the variance of
        the variational distibution over the cluster means q*(mu_k)
        - Ignoring constant terms!
        -------
        s2: K x D
        """
        s2 = 0
        #######################################################################
        # TODO: Implement the update for the variance of q*(mu_k) here using  #
        #       the appropriate class variables.                              #
        #######################################################################
        s2 = np.zeros((self.K,self.D))
        ones = np.ones((1, self.D))
        for i in range(self.K):
            sum_pi = 0
            for j in range(self.N):
                sum_pi +=self.pi[j, i]
            s2[i,:] = 1/(sum_pi*ones + 1/self.pmu_var[0])
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        return s2

    def update_pi(self):
        """
        Returns the *normalised* optimal factor for the probabilities
        of the mixture components q(c_i) - Ignoring constant terms!
        -------
        pi: N x K
        """
        pi = 0
        #######################################################################
        # TODO: Implement the update for the probabilities                    #
        #       of q(c_i). Return the normalised distribution.                #
        #######################################################################
        pi = np.zeros((self.N,self.K))
        for i in range(self.K):
            for j in range(self.N):
                T_ij = (-2 * self.X[j, :].reshape(self.D, 1).T.dot(self.m[i, :].reshape(self.D, 1)) + self.m[i,:].reshape(
                    self.D, 1).T.dot(self.m[i, :].reshape(self.D, 1)) + np.sum(np.diag(self.s2[i, :])))
                pi[j,i] = np.exp(-0.5*T_ij - 1)
        for i in range(self.N):
            pi[i] = pi[i] / np.sum(pi[i])
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        return pi

    def log_joint(self):

        """
        Returns the expected log probability density of the joint model
        log p(X, mu, c) under q(mu, c) - Ignoring constant terms!
        -------
        log_density: scalar
        """
        log_density = 0
        #######################################################################
        # TODO: Implement the expectation of the log density of the model     #
        #       here and return the sum - ignoring any constant terms         #
        #######################################################################

        log_density = 0
        for i in range(self.K):
            for j in range(self.N):
                log_density += self.pi[j,i]*(-2*self.X[j, :].reshape(self.D,1).T.dot(self.m[i, :].reshape(self.D,1)) + self.m[i, :].reshape(self.D,1).T.dot(self.m[i, :].reshape(self.D,1))+ np.sum(np.diag(self.s2[i, :])))
            log_density += (1/self.pmu_var[0])*(self.m[i, :].reshape(self.D,1).T.dot(self.m[i, :].reshape(self.D,1)) + np.sum(np.diag(self.s2[i, :])))
        log_density *= -0.5

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return log_density.item()

    def log_var(self):

        """
        Returns the expected log probability density of the variational distribution
        log q(mu, c) under q(mu, c) - Ignoring constant terms!
        -------
        log_q_density: scalar
        """
        log_q_density = 0
        #######################################################################
        # TODO: Implement the expectation of the log variational density here #
        #       and return the sum - ignoring any constant terms              #
        #######################################################################

        for i in range(self.K):
            for j in range(self.N):
                log_q_density += self.pi[j,i]*np.log(self.pi[j,i])
            log_q_density += -0.5*np.log(np.linalg.det(np.diag(self.s2[i, :])))

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        return log_q_density.item()

    def elbo(self):
        """
        Returns the evidence lower bound (ELBO) using log_joint and log_var
        -------
        elbo: scalar
        """
        elbo = 0
        #######################################################################
        # TODO: Implement the elbo using log_joint and log_var from before    #
        #######################################################################
        elbo += self.log_joint() - self.log_var()
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return elbo

    def cavi(self):
        """
        Updates variational parameters
        Do not change this function.
        """
        self.pi = self.update_pi()
        assert self.pi is not None, "Update for pi not implemented"
        self.s2 = self.update_s2()
        assert self.s2 is not None, "Update for s2 not implemented"
        self.m = self.update_m()
        assert self.m is not None, "Update for m not implemented"

    def fit(self, max_iter=1000, threshold=1e-10):
        """
        Performs CAVI using the optimal factor updates.
        Do not change this function.
        """

        elbo_trace = []
        for iter in range(max_iter):
            self.cavi()
            elbo_trace.append(self.elbo())

            if iter > 0:
                delta_elbo = elbo_trace[-1] - elbo_trace[-2]
                if delta_elbo < threshold:
                    break

        return np.float64(elbo_trace)
# mu = np.float64([
#         [-3., -3.],
#         [1., -3.],
#         [0., 2.],
#         [0.5, 3.],
#         [3., -1.]])
#
# K = mu.shape[0]
# X, C = generate_data(mu=mu) # X are the data, C are the cluster assignments (for visualisation)
# N, D = X.shape
# rng = np.random.RandomState(31440)
#
# pmu_var = np.float64(D * [1.])
#
# m = np.zeros(
# (K, D)) + rng.randn(K, D) * 1e-2
# s2 = np.ones_like(m)
#
# pi = rng.dirichlet(rng.randint(1, 10, size=K),N)
# rng = np.random.RandomState(31440)
#
# m = np.zeros(
#     (K, D)) + rng.randn(K, D) * 1e-2
# s2 = np.ones_like(m)
#
# X, C = generate_data(mu=mu)
# log_q_density = 0
#
# for i in range(K):
#     inv_diag_s = np.linalg.inv(np.diag(s2[i, :])) *np.linalg.inv(np.diag(s2[i, :]))
#     print(inv_diag_s.shape,K,D)
#     for j in range(N):
#         log_q_density += pi[j, i] * np.log(pi[j, i])
#     log_q_density += -0.5 * np.sum(inv_diag_s) \
#                      + m[i, :].reshape(D, 1).T.dot(np.diag(s2[i, :])).dot(m[i, :].reshape(D, 1)) + \
#                       m[i, :].reshape(D, 1).T.dot(np.diag(s2[i, :])).dot(m[i, :].reshape(D, 1))
#
# for i in range(K):
#     inv_diag_s = np.linalg.inv(np.diag(s2[i, :]))
#     print(inv_diag_s.shape,K,D)
#     for j in range(N):
#         log_q_density += pi[j, i] * np.log(pi[j, i])
#     log_q_density += -0.5 * np.sum(inv_diag_s) * np.sum(np.diag(s2[i, :])) \
#                      + m[i, :].reshape(D, 1).T.dot(m[i, :].reshape(D, 1)) + \
#                      np.sum(inv_diag_s) * m[i, :].reshape(D, 1).T.dot(m[i, :].reshape(D, 1))
# print(log_q_density.item())
# log_density = 0
# for i in range(K):
#     for j in range(N):
#         log_density += pi[j,i]*(-2*X[j, :].reshape(D,1).T.dot(m[i, :].reshape(D,1)) + m[i, :].reshape(D,1).T.dot(m[i, :].reshape(D,1))+ np.sum(np.diag(s2[i, :])))
#     log_density += np.linalg.inv(pmu_var[0]*np.identity(D))[0][0]*(m[i, :].reshape(D,1).T.dot(m[i, :].reshape(D,1)) + np.sum(np.diag(s2[i, :])))
# log_density *= -0.5
# print("1",log_density.item())
# print(np.diag(s2[0]))
# print(np.random.multivariate_normal(m[0],np.diag(s2[0])))
# print(mu_k[0].reshape(3,1).T.shape)
# mu_k = np.zeros((K,D))
# for k in range(K):
#     mu_k[k] = (np.random.multivariate_normal(m[k], np.diag(s2[k]))).T.reshape(1, D)
#     # print(mu_k[k])
# # print(mu_k)
# # print(pi)
# argmax =np.argmax(pi, axis=1)
# c = np.eye(N=N, M=K)[argmax]
# # print(argmax)
#
# # print(c)
# print(pi)
# sampled_c = np.random.choice(np.arange(K), 1 , p=pi[2,:])
# print(sampled_c)