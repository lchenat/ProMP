from meta_policy_search.policies.networks.mlp import create_mlp, forward_mlp
from meta_policy_search.policies.distributions.categorical import Categorical
from meta_policy_search.policies.base import Policy
from meta_policy_search.utils import Serializable, logger
from meta_policy_search.utils.utils import remove_scope_from_name

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class CategoricalMLPPolicy(Policy):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        Policy.__init__(self, *args, **kwargs)
        self.output_nonlinearlity = tf.nn.softmax

        self.init_policy = None
        self.policy_params = None
        self.obs_var = None
        self.prob_var = None
        self.action_var = None
        self._dist = None
        self.discrete = True

        self.build_graph()

    def build_graph(self):
        with tf.variable_scope(self.name):
            # build the actual policy network
            self.obs_var, self.prob_var = create_mlp(
                name='prob_network',
                output_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
                hidden_nonlinearity=self.hidden_nonlinearity,
                output_nonlinearity=self.output_nonlinearity,
                input_dim=(None, self.obs_dim,)
            )

            # symbolically define sampled action and distribution
            self.action_var = tf.random.categorical(tf.log(self.prob_var), 1)
            self._dist = Categorical(self.action_dim)

            # save the policy's trainable variables in dicts
            current_scope = tf.get_default_graph().get_name_scope()
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.policy_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])

    def get_action(self, observation):
        observation = np.expand_dims(observation, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[0], dict(prob=agent_infos['prob'][0])
        return action, agent_infos

    def get_actions(self, observations):
        assert observations.ndim == 2 and observations.shape[1] == self.obs_dim

        sess = tf.get_default_session()
        actions, probs = sess.run([self.action_var, self.prob_var],
            feed_dict={self.obs_var: observations})

        assert actions.shape == (observations.shape[0], self.action_dim)
        return actions, dict(prob=prob)

    def log_diagnostics(self, paths, prefix=''):
        pass

    def load_params(self, policy_params):
        raise NotImplementedError

    @property
    def distribution(self):
        return self._dist

    def distribution_info_sym(self, obs_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (dict) : a dictionary of placeholders or vars with the parameters of the MLP

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        if params is None:
            with tf.variable_scope(self.name):
                obs_var, prob_var = create_mlp(
                    name='prob_network',
                    output_dim=self.action_dim,
                    hidden_sizes=self.hidden_sizes,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    output_nonlinearity=self.output_nonlinearity,
                    input_var=obs_var,
                    reuse=True,
                )
        else:
            prob_network_params = OrderedDict()
            for name, param in params.items():
                prob_network_params[name] = param

            obs_var, prob_var = forward_mlp(
                output_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
                hidden_nonlinearity=self.hidden_nonlinearity,
                output_nonlinearity=self.output_nonlinearity,
                input_var=obs_var,
                mlp_params=prob_network_params,
            )

        return dict(prob=prob_var)

    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise ['prob']

