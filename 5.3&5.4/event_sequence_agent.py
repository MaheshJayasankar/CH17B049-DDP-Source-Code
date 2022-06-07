# Simplified RL Agent set-up for event sequence optimization

import numpy as np

class RlAgent:
    def __init__(self, state_space, action_space, epsilon = 0.3, epsilon_decay = 0, learning_rate = 0.1, seed = 0):
        rand_state = np.random.get_state()
        np.random.seed(seed)
        self.Q = np.random.random_sample((*state_space, action_space))
        self.num_states = action_space
        self.epsilon = epsilon
        self.eps_dec = epsilon_decay
        self.rand_state = np.random.get_state()
        self.alpha = learning_rate
        np.random.set_state(rand_state)
    
    def pick_action(self, state):
        glob_rand_state = np.random.get_state()
        np.random.set_state(self.rand_state)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_states)
        else:
            action = np.argmax(self.Q[state])
        self.rand_state = np.random.get_state()
        np.random.set_state(glob_rand_state)
        self.epsilon *= (1 - self.eps_dec)
        return action
    
    def update(self, state, action, reward, next_state):
        idx = (*state, action)
        old_q_value = self.Q[idx]

        next_q_values = self.Q[next_state]

        max_q = np.max(next_q_values)

        new_q_value = reward + max_q

        self.Q[idx] = old_q_value + self.alpha * (new_q_value - old_q_value)
