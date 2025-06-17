from spinup.utils.run_utils import ExperimentGrid
from RLMultilayer.algos.ppo.ppo import ppo
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.tasks import get_env_fn
import torch
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--env', type=str, default='erent_1500_absorber_task_v0')
    parser.add_argument('--exp_name', type=str, default='selective_1500nm_absorber')
    parser.add_argument('--discrete_thick', action="store_true")
    parser.add_argument('--maxlen', default=5, type=int)
    parser.add_argument('--hierarchical', action='store_true', help='if set to true, then output out the material type first, then condition the material thickness on the material type')
    parser.add_argument('--use_rnn', action='store_true')
    parser.add_argument('--spectrum_repr', action='store_true')
    parser.add_argument('--not_repeat', action='store_true', help='if set to true, don\'t repeat materials')
    args = parser.parse_args()

    # Environment kwargs - DO NOT override merit_func since our task has the correct one built-in
    env_kwargs = {
        "discrete_thick": args.discrete_thick, 
        'spectrum_repr': args.spectrum_repr, 
        'bottom_up': False
        # Removed merit_func override - our task already has cal_reward_selective_1500
    }

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', get_env_fn(args.env, **env_kwargs))
    eg.add('seed', [42*(i+1) for i in range(args.num_runs)])
    eg.add('epochs', 50)  # Set to 50 epochs as requested
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(64,)], 'hid')
    eg.add('ac_kwargs:cell_size', 64, '')
    # eg.add('ac_kwargs:not_repeat', [True, False])
    eg.add('ac_kwargs:not_repeat', [args.not_repeat])
    eg.add('ac_kwargs:ortho_init', ['on'])
    # eg.add('ac_kwargs:hierarchical', [True, False])
    eg.add('ac_kwargs:hierarchical', [args.hierarchical])
    eg.add('ac_kwargs:channels', 16)
    eg.add('ac_kwargs:act_emb', [True])
    eg.add('ac_kwargs:act_emb_dim', 5)
    eg.add('use_rnn', args.use_rnn)
    eg.add('gamma', 1)
    eg.add('beta', [0.01])
    eg.add('lam', [0.95])
    eg.add('max_ep_len', [args.maxlen], in_name=True)
    eg.add('actor_critic', core.RNNActorCritic if args.use_rnn else core.MLPActorCritic)
    eg.add("train_pi_iters", [5])
    eg.add("pi_lr", [5e-5])
    eg.add('reward_factor', [1])
    eg.add('spectrum_repr', [args.spectrum_repr])
    eg.add('ac_kwargs:scalar_thick', [False])

    print("="*60)
    print("TRAINING 1500nm SELECTIVE ABSORBER")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Epochs: 50")
    print(f"Steps per epoch: 1000") 
    print(f"Max episode length: {args.maxlen}")
    print(f"Discrete thickness: {args.discrete_thick}")
    print(f"Experiment name: {args.exp_name}")
    print("="*60)

    eg.run(ppo, num_cpu=args.cpu, data_dir='./Experiments/{}'.format(args.exp_name), datestamp=False)