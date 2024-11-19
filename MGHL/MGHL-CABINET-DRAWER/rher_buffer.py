from utils.replay_buffer import ReplayBuffer, Trajectory
import numpy as np
import torch
from copy import deepcopy as dc


class RHERReplayBuffer(ReplayBuffer):
    """ Hindisght Experience Replay Buffer """

    def __init__(self, 
                 env, 
                 capacity: int, 
                 k_future: int,
                 sub_eps: float) -> None:
        super().__init__(capacity)
        self.env = env  # 需要调用 compute_reward 函数
        self.future_p = 1 - (1. / (1 + k_future))
        self.sub_eps = sub_eps

    def push(self, trajectory: Trajectory):
        """ 存储 trajectory 到经验回放中

            :param trajectory: (Trajectory)
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int = 256, sequential: bool = True, with_log=True, device='cpu', task='reach'):
        """ 从经验回放中随机采样 batch_size 个 transitions

        :param batch_size: (int) 采样的 batch size
        :param sequential: (bool)
        :param with_log: (bool)
        :param device: (str) cpu or cuda
        :param task: (str) reach, drawer, place
        :return: (torch.tensor) s, a, s_, r, g
        """
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        time_indices = np.random.randint(0, [len(self.buffer[episode]) for episode in ep_indices])

        states = np.array([self.buffer[episode].buffer[timestep][0].copy() for episode, timestep in zip(ep_indices, time_indices)])
        actions = np.array([self.buffer[episode].buffer[timestep][1].copy() for episode, timestep in zip(ep_indices, time_indices)])
        next_states = np.array([self.buffer[episode].buffer[timestep][2].copy() for episode, timestep in zip(ep_indices, time_indices)])
        achieved_goals = np.array([self.buffer[episode].buffer[timestep][4].copy() for episode, timestep in zip(ep_indices, time_indices)])
        desired_goals = np.array([self.buffer[episode].buffer[timestep][5].copy() for episode, timestep in zip(ep_indices, time_indices)])

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = []
        for episode, timestep in zip(ep_indices, time_indices):
            future_offset.append(np.random.randint(timestep, len(self.buffer[episode])))
        future_offset = np.array(future_offset).astype(int)
        future_t = future_offset[her_indices]

        if task == 'reach':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][:3]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, :3] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, :3], desired_goals[:, :3], th=self.sub_eps), 1)
            desired_goals[:, 3:] *= 0
        elif task == 'unlock':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:6]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:6] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:6], desired_goals[:, 3:6], th=0.02), 1)
            desired_goals[:, :3] *= 0
            desired_goals[:, 6:] *= 0
        elif task == 'door':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:9]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:9] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:9], desired_goals[:, 3:9], th=0.02), 1)
            desired_goals[:, :6] *= 0
            desired_goals[:, 9:] *= 0
        elif task == 'drawer':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:12]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:12] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:12], desired_goals[:, 3:12], th=0.02), 1)
            desired_goals[:, :9] *= 0
            desired_goals[:, 12:] *= 0
        elif task == 'place':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:15]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:15] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:15], desired_goals[:, 3:15], th=0.02), 1)
            desired_goals[:, :12] *= 0

        s = torch.tensor(states, dtype=torch.float).to(device)
        a = torch.tensor(actions, dtype=torch.float).to(device)
        s_ = torch.tensor(next_states, dtype=torch.float).to(device)
        r = torch.tensor(rewards, dtype=torch.float).to(device)
        g = torch.tensor(desired_goals, dtype=torch.float).to(device)

        return s, a, s_, r, g
