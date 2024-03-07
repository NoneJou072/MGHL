import os
import time
import argparse

import torch
from robopal.demos.manipulation_tasks.demo_cabinet import LockedCabinetEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper
from tqdm import trange

from utils.ModelBase import ModelBase
from iorl import IORL

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, './log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for RHERTD3")
    parser.add_argument("--env_name", type=str, default="LockedCabinet-v1", help="env name")
    parser.add_argument("--algo_name", type=str, default="HDTO", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    # Training Params
    parser.add_argument("--max_test_episodes", type=int, default=int(1e2), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=2e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Reply buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    # Net Params
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of QNet")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="Softly update the target network")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick: state normalization")
    parser.add_argument("--random_steps", type=int, default=1e3,
                        help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--update_freq", type=int, default=40, help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")
    parser.add_argument("--k_update", type=bool, default=2, help="Delayed policy update frequence")

    return parser.parse_args()


class HERDDPGModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = IORL(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{6}_seed_{self.args.seed}'
        self.load_weights()

    def load_weights(self):
        model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
        model_path = os.path.join(model_dir, f'{self.model_name}.pth')
        actor_state_dict = torch.load(model_path)
        self.agent.actor.load_state_dict(actor_state_dict)

    def play(self):
        success_count = 0
        for ep in trange(self.args.max_test_episodes):
            self.env.env.TASK_FLAG = 1
            task = 'door'
            obs, info = self.env.reset()
            for _ in range(self.env.env.max_episode_steps):

                # if info['is_unlock_success'] == 1.0:
                #     task = 'door'

                a = self.agent.sample_action(obs, task=task, deterministic=True)
                obs, r, terminated, truncated, info = self.env.step(a)

            success_count += info['is_door_success']

        print(success_count)
        self.env.close()


def make_env(args):
    """ 配置环境 """
    env = LockedCabinetEnv(render_mode=None)
    env = GoalEnvWrapper(env)

    setattr(args, 'state_dim', env.observation_space.spaces["observation"].shape[0])
    setattr(args, 'action_dim', env.action_space.shape[0])
    setattr(args, 'goal_dim', env.observation_space.spaces["desired_goal"].shape[0])
    setattr(args, 'max_episode_steps', env.max_episode_steps)
    setattr(args, 'min_action', float(env.action_space.low[0]))
    setattr(args, 'max_action', float(env.action_space.high[0]))

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = HERDDPGModel(
        env=env,
        args=args,
    )
    model.play()
