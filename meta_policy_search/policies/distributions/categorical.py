import tensorflow as tf
import numpy as np
from meta_policy_search.policies.distributions.base import Distribution

TINY = 1e-8


def log_softmax(x):
    x -= x.max(-1, keepdim=True)
    return x - np.log(np.exp(x).sum(-1, keepdim=True) + TINY)


def softmax(x):
    x -= x.max(-1, keepdim=True)
    expx = np.exp(x)
    return expx / expx.sum(-1, keepdim=True)


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """
        old_prob = old_dist_info_vars["prob"]
        new_prob = new_dist_info_vars["prob"]
        old_logprob = tf.log(old_prob + TINY)
        new_logprob = tf.log(new_prob + TINY)

        # assert ranks
        tf.assert_rank(old_prob, 2)
        tf.assert_rank(new_prob, 2)

        return tf.reduce_sum(
            old_prob * (old_logprob - new_logprob), reduction_indices=-1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices

       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """

        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]
        old_logprob = np.log(old_prob + TINY)
        new_logprob = np.log(old_prob + TINY)


        # assert ranks
        assert old_prob.ndim == 2
        assert new_prob.ndim == 2

        return np.sum(
            old_prob * (old_logprob - new_logprob), axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (tf.Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): likelihood ratio
        """
        with tf.variable_scope("log_li_new"):
            logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        with tf.variable_scope("log_li_old"):
            logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        log_probs = tf.log(probs + TINY)

        # assert ranks
        tf.assert_rank(probs, 2)
        tf.assert_rank(x_var, 2)

        return tf.batch_gather(log_probs, x_var)

    def log_likelihood(self, xs, dist_info):
        assert xs.ndim == 1
        probs = dist_info["prob"]

        return np.log(probs + TINY)[np.arange(len(xs)), xs]

    def entropy_sym(self, dist_info_vars):
        probs = dist_info_vars["prob"]
        log_probs = tf.log(probs + TINY)
        return -tf.reduce_sum(probs * log_probs, reduction_indices=-1)

    def entropy(self, dist_info):
        probs = dist_info["prob"]
        log_probs = np.log(probs + TINY)
        return -np.sum(probs * log_probs, axis=-1)

    def sample(self, dist_info):
        probs = dist_info["prob"]
        return np.asarray([np.random.choice(prob.shape[1], p=prob) for prob in probs])

    @property
    def dist_info_specs(self):
        return [("prob", (self.dim,))]
