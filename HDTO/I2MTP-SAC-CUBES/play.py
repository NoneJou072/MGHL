import os
import argparse

import numpy as np
import torch
from tqdm import trange

from utils.ModelBase import ModelBase
from iorl import IORL

from robopal.demos.manipulation_tasks.demo_multi_cubes import MultiCubes
from robopal.commons.gym_wrapper import GoalEnvWrapper

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, './log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for HDTO")
    parser.add_argument("--env_name", type=str, default="MultiCubeStack-v1", help="env name")
    parser.add_argument("--algo_name", type=str, default="HDTO", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    # Training Params
    parser.add_argument("--max_test_episodes", type=int, default=int(1e2), help=" Maximum number of training steps")
    parser.add_argument("--update_freq", type=int, default=150, help="Take 150 steps,then update the networks 50 times")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Reply buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    # Net Params
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of QNet")
    parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="Softly update the target network")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")

    return parser.parse_args()


class HERDDPGModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = IORL(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{6}_seed_10'
        # self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{6}_seed_{self.args.seed}'
        self.load_weights()

    def load_weights(self):
        model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
        model_path = os.path.join(model_dir, f'{self.model_name}.pth')
        actor_state_dict = torch.load(model_path)
        self.agent.actor.load_state_dict(actor_state_dict)

    def play(self):
        success = 0
        for ep in trange(self.args.max_test_episodes):
            self.env.env.task = 'blue'
            obs, info = self.env.reset()
            for t in range(50):

                # if info['is_red_success'] == 1.0 and self.env.env.task == 'red':
                #     for _ in range(10):
                #         a = np.array([0.0, 0.0, 1.0, 1.0])
                #         obs, r, terminated, truncated, info = self.env.step(a)
                #     self.env.env.task = 'green'
                #
                # if info['is_red_success'] == 1.0 and info['is_green_success'] == 1.0 and self.env.env.task == 'green':
                #     for _ in range(10):
                #         a = np.array([0.0, 0.0, 1.0, 1.0])
                #         obs, r, terminated, truncated, info = self.env.step(a)
                #     self.env.env.task = 'blue'

                a = self.agent.sample_action(obs, task=self.env.env.task, deterministic=True)
                obs, r, terminated, truncated, info = self.env.step(a)
            if info['is_red_success'] == 1.0 and info['is_green_success'] == 1.0 and info['is_blue_success'] == 1.0:
                success += 1
        print(success)
        self.env.close()


def make_env(args):
    """ 配置环境 """
    env = MultiCubes(render_mode='human')
    env = GoalEnvWrapper(env)
    state_dim = env.observation_space.spaces["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'goal_dim', goal_dim)
    setattr(args, 'max_episode_steps', env.max_episode_steps)
    setattr(args, 'max_action', max_action)

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = HERDDPGModel(
        env=env,
        args=args,
    )
    model.play()
