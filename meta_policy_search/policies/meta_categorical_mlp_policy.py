from meta_policy_search.policies.base import MetaPolicy
from meta_policy_search.policies.categorical_mlp_policy import CategoricalMLPPolicy
import numpy as np
import tensorflow as tf
from meta_policy_search.policies.networks.mlp import forward_mlp


class MetaCategoricalMLPPolicy(CategoricalMLPPolicy, MetaPolicy):
    def __init__(self, meta_batch_size,  *args, **kwargs):
        self.quick_init(locals()) # store init arguments for serialization
        self.meta_batch_size = meta_batch_size

        self.pre_update_action_var = None
        self.pre_update_prob_var = None

        self.post_update_action_var = None
        self.post_update_prob_var = None

        super(MetaCategoricalMLPPolicy, self).__init__(*args, **kwargs)

    def build_graph(self):
        """
        Builds computational graph for policy
        """

        # Create pre-update policy by calling build_graph of the super class
        super(MetaCategoricalMLPPolicy, self).build_graph()
        self.pre_update_action_var = tf.split(self.action_var, self.meta_batch_size)
        self.pre_update_prob_var = tf.split(self.prob_var, self.meta_batch_size)

        # Create lightweight policy graph that takes the policy parameters as placeholders
        with tf.variable_scope(self.name + "_ph_graph"):
            prob_network_phs_meta_batch = []

            self.post_update_action_var = []
            self.post_update_prob_var = []

            # build meta_batch_size graphs for post-update policies --> thereby the policy parameters are placeholders
            obs_var_per_task = tf.split(self.obs_var, self.meta_batch_size, axis=0)

            for idx in range(self.meta_batch_size):
                with tf.variable_scope("task_%i" % idx):

                    with tf.variable_scope("prob_network"):
                        # create prob network parameter placeholders
                        prob_network_phs = self._create_placeholders_for_vars(
                            scope=self.name + "/prob_network")  # -> returns ordered dict
                        prob_network_phs_meta_batch.append(prob_network_phs)

                        # forward pass through the prob mlp
                        _, prob_var = forward_mlp(
                            output_dim=self.action_dim,
                            hidden_sizes=self.hidden_sizes,
                            hidden_nonlinearity=self.hidden_nonlinearity,
                            output_nonlinearity=self.output_nonlinearity,
                            input_var=obs_var_per_task[idx],
                            mlp_params=prob_network_phs,
                        )

                    action_var = tf.random.categorical(tf.log(prob_var), 1)

                    self.post_update_action_var.append(action_var)
                    self.post_update_prob_var.append(prob_var)

            self.policies_params_phs = []
            for idx, odict in enumerate(prob_network_phs_meta_batch):
                self.policies_params_phs.append(odict)

            self.policy_params_keys = list(self.policies_params_phs[0].keys())

    def get_action(self, observation, task=0):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.repeat(np.expand_dims(np.expand_dims(observation, axis=0), axis=0), self.meta_batch_size, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[task][0], dict(prob=agent_infos[task][0]['prob'])
        return action, agent_infos

    def get_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        Returns:
            (tuple) : A tuple containing a list of numpy arrays of action, and a list of list of dicts of agent infos
        """
        assert len(observations) == self.meta_batch_size

        if self._pre_update_mode:
            actions, agent_infos = self._get_pre_update_actions(observations)
        else:
            actions, agent_infos = self._get_post_update_actions(observations)


        assert len(actions) == self.meta_batch_size
        return actions, agent_infos

    def _get_pre_update_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        """
        batch_size = observations[0].shape[0]
        assert all([obs.shape[0] == batch_size for obs in observations])
        assert len(observations) == self.meta_batch_size
        obs_stack = np.concatenate(observations, axis=0)
        feed_dict = {self.obs_var: obs_stack}

        sess = tf.get_default_session()
        actions, probs = sess.run([self.pre_update_action_var,
                                             self.pre_update_prob_var],
                                            feed_dict=feed_dict)
        agent_infos = [[dict(prob=prob) for prob in probs[idx]] for idx in range(self.meta_batch_size)]
        return actions, agent_infos

    def _get_post_update_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        """
        assert self.policies_params_vals is not None
        obs_stack = np.concatenate(observations, axis=0)
        feed_dict = {self.obs_var: obs_stack}
        feed_dict.update(self.policies_params_feed_dict)

        sess = tf.get_default_session()
        actions, probs = sess.run([self.post_update_action_var,
                                             self.post_update_prob_var],
                                            feed_dict=feed_dict)
        agent_infos = [[dict(prob=prob) for prob in probs[idx]] for idx in range(self.meta_batch_size)]
        return actions, agent_infos

