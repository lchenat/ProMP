import _init_path
from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_categorical_mlp_policy import MetaCategoricalMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder
from meta_policy_search.envs.goal_world.reach_env import MetaReachWorld
from meta_policy_search.envs.goal_world.wrapper_env import DistanceGoalRewardWrapper

import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time
from ipdb import launch_ipdb_on_exception

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rollouts_per_meta_task', type=int, default=20)
    parser.add_argument('max_path_length', type=int, default=50)
    parser.add_argument('--seq', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=1)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--inner_lr', type=float, default=0.1)  # adaptation step size
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # meta-policy gradient step size
    parser.add_argument('--num_promp_steps', type=int, default=5)  # number of ProMp steps without re-sampling
    parser.add_argument('--clip_eps', type=float, default=0.3)  # clipping range
    parser.add_argument('--target_inner_step', type=float, default=0.01)
    parser.add_argument('--init_inner_kl_penalty', type=float, default=5e-4)
    parser.add_argument('--adaptive_inner_kl_penalty', action='store_true')  # whether to use an adaptive or fixed KL-penalty coefficient
    parser.add_argument('--n_itr', type=int, default=1001)  # number of overall training iterations
    parser.add_argument('--meta_batch_size', type=int, default=40)  # number of sampled meta-tasks per iterations
    parser.add_argument('--num_inner_grad_steps', type=int, default=1)  # number of inner / adaptation gradient steps
    parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)
    return parser.parse_args(args)

def get_env():
    return DistanceGoalRewardWrapper(
        MetaReachWorld(
            map_name='map6x6',
            tasks=[
                (1, 1),
                (1, 6),
                (6, 1),
                (6, 6),
            ],
        )
    )

def main(args=None):
    idx = int(time.time())
    args = parse_args(args)

    config = {
        'seed': args.seed,
        'baseline': 'LinearFeatureBaseline',
        'env': 'ReachWorld', # not used
        'rollouts_per_meta_task': args.rollout_per_meta_task,
        'max_path_length': args.max_path_length, # 100
        'parallel': not args.seq,
        'discount': args.discount,
        'gae_lambda': args.gae_lambda,
        'normalize_adv': True,
        'hidden_sizes': args.hidden_sizes,
        'inner_lr': args.inner_lr, # adaptation step size
        'learning_rate': args.learning_rate, # meta-policy gradient step size
        'num_promp_steps': args.num_promp_steps, # number of ProMp steps without re-sampling
        'clip_eps': args.clip_eps, # clipping range
        'target_inner_step': args.target_inner_step,
        'init_inner_kl_penalty': args.init_inner_kl_penalty,
        'adaptive_inner_kl_penalty': args.adaptive_inner_kl_penalty, # whether to use an adaptive or fixed KL-penalty coefficient
        'n_itr': args.n_itr, # number of overall training iterations
        'meta_batch_size': args.meta_batch_size, # number of sampled meta-tasks per iterations
        'num_inner_grad_steps': args.num_inner_grad_steps, # number of inner / adaptation gradient steps

    }

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    set_seed(config['seed'])


    baseline =  globals()[config['baseline']]() #instantiate baseline

    env = get_env()
    #env = normalize(env) # apply normalize wrapper to env

    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = np.prod(env.action_space.shape)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise Exception('unknown action space, cannot get action dim')

    policy = MetaCategoricalMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=action_dim,
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
    )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer.train()


if __name__=="__main__":
    with launch_ipdb_on_exception():
        main()