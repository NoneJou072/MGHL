from Sarsa import Sarsa
from utils.ModelBase import ModelBase
import argparse
import gymnasium as gym


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for Sarsa")
    parser.add_argument("--env_name", type=str, default="CliffWalking-v0", help="env name")
    parser.add_argument("--algo_name", type=str, default="Sarsa", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cpu', help="pytorch device")
    # Training Params
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate of QNet")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=0.95, help="ε-贪心策略中的初始epsilon，减小此值可减少学习开始时的随机探索几率")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="ε-贪心策略中的终止epsilon，越小学习结果越逼近")
    parser.add_argument("--epsilon_decay", type=float, default=300, help="e-greedy策略中epsilon的衰减率，此值越大衰减的速度越快")

    return parser.parse_args()


class SarsaModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = Sarsa(args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{1}_seed_{self.args.seed}'

    def train(self):
        total_steps = 0
        rewards = []
        while total_steps < self.args.max_train_steps:
            ep_reward = 0
            state = env.reset()
            action = self.agent.sample_action(state) 
            while True:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = self.agent.sample_action(next_state)
                
                self.agent.update(state, action, reward, next_state, next_action, terminated)
                
                state = next_state
                action = next_action
                
                total_steps += 1
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            print("steps:{}/{}: reward:{:.1f}".format(total_steps + 1, self.args.max_train_steps, ep_reward))

        print("完成训练！")
        self.env.close()


def make_env(args):
    """ 配置环境 """
    env = gym.make('CliffWalking-v0')  # 定义环境

    state_dim = env.observation_space.n  # 状态数
    action_dim = env.action_space.n  # 动作数
    print(f"state dim:{state_dim}, action dim:{action_dim}")
    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = SarsaModel(
        env=env,
        args=args,
    )
    model.train()
