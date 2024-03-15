import os
import argparse
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from robopal.demos.manipulation_tasks.demo_cube_drawer import DrawerCubeEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper

from iorl import IORL
from utils.ModelBase import ModelBase
from utils.replay_buffer import Trajectory

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, './log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for HDTO")
    parser.add_argument("--env_name", type=str, default="DrawerBox-v1", help="env name")
    parser.add_argument("--algo_name", type=str, default="HDTO", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    # Training Params
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=2e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Reply buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    # Net Params
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of QNet")
    parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="Softly update the target network")
    parser.add_argument("--random_steps", type=int, default=1e3,
                        help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--update_freq", type=int, default=100, help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")
    parser.add_argument("--task_list", type=list, default=['drawer', 'place', 'reach'], help="task list")

    return parser.parse_args()


class IORLModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = IORL(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{4}_seed_{self.args.seed}'
        self.random_steps = args.random_steps
        self.update_freq = args.update_freq
        self.max_train_steps = args.max_train_steps

        self.best_total_success_rate = 0.0

    def train(self):
        """ 训练 """
        print("开始训练！")
        total_steps = 0
        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        if os.path.exists(log_dir):
            i = 1
            while os.path.exists(log_dir + '_' + str(i)):
                i += 1
            log_dir = log_dir + '_' + str(i)

        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.max_train_steps:
            # Task1
            self.env.env.TASK_FLAG = 0
            obs, _ = self.env.reset()
            traj_draw = Trajectory()
            while True:
                a = self.agent.sample_action(obs, task='drawer', deterministic=False)  # 选择动作
                obs_, r, terminated, truncated, info = self.env.step(a)  # 更新环境，返回transition
                traj_draw.push((obs["observation"], a, obs_["observation"], r, obs["achieved_goal"], obs["desired_goal"],
                               obs_["achieved_goal"]))
                obs = deepcopy(obs_)
                total_steps += 1
                if truncated:
                    break

            # Task2
            self.env.env.TASK_FLAG = 1
            obs, _ = self.env.reset()
            traj_place = Trajectory()
            while True:
                a = self.agent.sample_action(obs, task='place', deterministic=False)  # 选择动作
                obs_, r, terminated, truncated, info = self.env.step(a)  # 更新环境，返回transition
                traj_place.push((obs["observation"], a, obs_["observation"], r, obs["achieved_goal"],
                                obs["desired_goal"], obs_["achieved_goal"]))
                obs = deepcopy(obs_)
                total_steps += 1
                if truncated:
                    break

            self.agent.memory_draw.push(traj_draw)
            self.agent.memory_place.push(traj_place)

            if total_steps >= self.random_steps:
                for _ in range(self.update_freq):
                    self.agent.update()
                self.agent.update_target_net()

            if total_steps >= self.random_steps and total_steps % self.args.evaluate_freq == 0:
                success_rate_drawer, success_rate_place = self.evaluate_policy()
                print(
                    f"total_steps:{total_steps} \t  success_rate_drawer:{success_rate_drawer}\t \
                    success_rate_place:{success_rate_place}"
                )
                writer.add_scalar('success_rate/success_rate_{}_drawer'.format(self.args.env_name), success_rate_drawer,
                                  global_step=total_steps)
                writer.add_scalar('success_rate/success_rate_{}_place'.format(self.args.env_name), success_rate_place,
                                  global_step=total_steps)
                writer.add_scalar('loss/critic_loss_{}'.format(self.args.env_name), self.agent.critic_loss_record,
                                  global_step=total_steps)
                writer.add_scalar('loss/actor_loss_{}'.format(self.args.env_name), self.agent.actor_loss_record,
                                  global_step=total_steps)
                # Save Actor weights
                model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                total_success_rate = success_rate_drawer + success_rate_place
                if total_success_rate > self.best_total_success_rate:
                    self.best_total_success_rate = total_success_rate
                    torch.save(self.agent.actor.state_dict(), os.path.join(model_dir, f'{self.model_name}.pth'))
        print("完成训练！")
        self.env.close()

    def evaluate_policy(self):
        times = 10
        # evaluate drawer stage
        self.env.env.TASK_FLAG = 0
        success_rate_drawer = 0
        for _ in range(times):
            episode_reward = 0
            obs, _ = self.env_evaluate.reset()
            while True:
                # We use the deterministic policy during the evaluating
                action = self.agent.sample_action(obs, task='drawer', deterministic=True)
                obs_, r, terminated, truncated, info = self.env_evaluate.step(action)
                obs = deepcopy(obs_)
                episode_reward += r
                if truncated:
                    success_rate_drawer += info['is_drawer_success']
                    break

        # evaluate place stage
        self.env.env.TASK_FLAG = 1
        success_rate_place = 0
        for _ in range(times):
            episode_reward = 0
            obs, _ = self.env_evaluate.reset()
            while True:
                action = self.agent.sample_action(obs, task='place', deterministic=True)
                obs, r, terminated, truncated, info = self.env_evaluate.step(action)
                episode_reward += r
                if truncated:
                    success_rate_place += info['is_place_success']
                    break

        return success_rate_drawer / times, success_rate_place / times


def make_env(args):
    """ 配置环境 """
    env = DrawerCubeEnv(render_mode='human')
    env = GoalEnvWrapper(env)

    state_dim = env.observation_space.spaces["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    print(f"state dim:{state_dim}, action dim:{action_dim}, max_epi_steps:{env.max_episode_steps}")
    print(f"max action:{max_action}, min action:{min_action}")

    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'goal_dim', goal_dim)
    setattr(args, 'max_episode_steps', env.max_episode_steps)
    setattr(args, 'max_action', max_action)

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = IORLModel(
        env=env,
        args=args,
    )
    model.train()
