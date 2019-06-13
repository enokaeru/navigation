import numpy as np
import random

from collections import namedtuple

from model import QNetworklow

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import MinSegmentTree, SumSegmentTree ,LinearSchedule

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                 lr=5e-4,
                 total_timesteps=1000000,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6
                 ):
        """Initialize an Agent object.

        Params
        ------
            state_size: int
                dimension of each state
            action_size: int
                dimension of each action
            seed: int
                random seed
            total_timesteps: int
                l
             prioritized_replay_alpha: float
                alpha parameter for prioritized replay buffer
             prioritized_replay_beta0: float
                initial value of beta for prioritized replay buffer
             prioritized_replay_beta_iters: int
                number of iterations over which beta will be annealed from initial value
                to 1.0. If set to None equals to total_timesteps.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetworklow(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetworklow(state_size, action_size, seed).to(device)
        # optimizer along reinbow paper
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                        initial_p=prioritized_replay_beta0,
                                        final_p=1.0)
        # Initialize time step(for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.prioritized_replay_eps=prioritized_replay_eps

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE, self.beta_schedule.value(self.t_step))
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """
         Returns actions for given state as per current policy
         Params
         ------
            state : array_like
                current state
            eps : float
                epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_value = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selction
        if random.random() > eps:
            return np.argmax(action_value.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # TODO: we should adapt priority replay buffer
    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Params
        ------
            experineces : Tuple[torch.Tensor]
                tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, done, weights, batch_idxes = experiences

        # Get max predicted Q values (for next states) from target model
        # modify to DDQN
        # Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        next_action = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_action)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        td_error = (Q_targets.cpu().detach() - Q_expected.cpu().detach()).abs().numpy()
        new_priorities = td_error + self.prioritized_replay_eps
        self.memory.update_priorities(batch_idxes.numpy(), new_priorities)



        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft uodate model paramters
        (ref: DDPG, https://arxiv.org/abs/1509.02971)
        Params
        -----
            local_model: Pytorch model
                weights will be copied from 
            target_model:
                weights will be copied to
            tau: float
                interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ------
            action_size: int
                dimension of each action
            buffer_size: int
                maximum size of buffer
            batch_size: int
                size of each training batch
            seed: int
                random seed
        """
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self._next_idx = 0
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        if self._next_idx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self._next_idx] = e
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory"""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).astype(np.uint8).float().to(
            device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay Buffer
    ref:https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """
        Initalize a Prioritized Replay Buffer
        Params
        ------
            action_size: int
                dimension of each action
            buffer_size: int
                maximum size of buffer
            batch_size: int
                size of each training batch
            seed: int
                random seed
            alpha: float
                how much prioritization is used
                (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        # adapt data stractures to semenget tree structure.
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        # set max priority (be sure to be learned once)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        # generate uniformly random value's indexes from 0 to (p_total/batch_size)
        # sample experiences from the list
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """
        Sample a batch of experiences.
        
        Params
        ------
            batch_size: int
                How many transitions to sample
            beta: float
                To what degree to use importance weights
                (0 - no corrections, 1 - full correction)

        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        # calculate the weight
        weights = []
        # calculate max_weight for normalization of the weight
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        encoded_sample = self._encode_sample(idxes)
        weights = torch.from_numpy(np.array(weights)).float().to(device)
        idxes = torch.from_numpy(np.array(idxes)).int().to(device)

        return tuple(list(encoded_sample) + [weights, idxes])

    def _encode_sample(self, idxes):
        experiences = [self.memory[idx] for idx in idxes]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return (states, actions, rewards, next_states, dones)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        set priority of transition at index idxes[i] in buffer
        
        Params
        ------
        idexes: [int]
            List of indexs of sampled transitions
        priorities: [float]
            List of update priorities corresponding to 
            transitions at the sampled idxes denoted by
            variable 'idxes'
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


