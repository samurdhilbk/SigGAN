from SigGAN.models import Generator, GeneratorPretraining, Discriminator
import keras.backend as K
import numpy as np

class Agent(object):
    def __init__(self, sess, B, H, lr=1e-3):
        self.sess = sess
        self.B = B
        self.H = H
        self.lr = lr
        self.generator = Generator(sess, H, lr)

    def act(self, state):
        action = self.generator.predict(state)
        return action

#     def reset(self):
#         self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment(object):
    def __init__(self, rf_system, discriminator, B, T, g_beta, n_sample=16):
        self.rf_system = rf_system
        self.B = B
        self.T = T
        self.n_sample = n_sample
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.symbols = None
        self.reward = None
        self.result = None
        self._state = None
        #self.reset()

    def get_state(self):
        return self._state
    
    def get_result(self):
        return self.result

    def reset(self, symbols):
        self.t = 0
        self.reward = None
        self.result = None
        self.symbols = symbols
        self._state = self.symbols[0].reshape(1,1,2)

    def process_reward(self, perturbed_signal):
        trans_signal = self.rf_system.transmit_real_symbol_block(perturbed_signal.reshape(self.T,2), impersonator=True)
        return (self.discriminator.predict(trans_signal.reshape(-1,self.T,2)).squeeze()) - 0.5
    
    def step(self, action):
        #print(self.t)
        self._update_state(action)
        next_state = self.get_state()
        
        reward = self.Q(action, self.n_sample)
        #print("RRReward ", reward)
        info = None
        
        self.t = self.t + 1
        
        is_episode_end = self.t >= self.T

        return [next_state, reward, is_episode_end, info]

    
    def Q(self, action, n_sample=16):
        if self.reward is not None: return self.reward
        reward = 0
        Y_base = self.get_state()    # (B, t-1)
        R_base = self.get_result()
        
        if (self.t+1) >= self.T:
            Y = self._update_state(action, current_state=Y_base, tau=self.t)
            return self.process_reward(self.get_result())

        # Rollout
        for idx_sample in range(n_sample):
            #print(idx_sample)
            Y = Y_base
            R = R_base
            for tau in range(self.t+1, self.T):
                y_tau = self.g_beta.act(Y)
                Y = self._update_state(y_tau, current_state=Y, tau=tau)
                R = self._update_result(y_tau, current_result=R)

            reward += self.process_reward(R) / n_sample
        
        self.reward=reward
        return self.reward
    
#     def Q(self, action, n_sample=16):
#         reward = 0
#         Y_base = self.get_state()    # (B, t-1)

#         if self.t >= self.T:
#             Y = self._update_state(action, current_state=Y_base)
#             return self.process_rewardpredict(Y)

#         # Rollout
#         for idx_sample in range(n_sample):
#             print(idx_sample)
#             Y = Y_base
#             y_t = self.g_beta.act(Y)
#             Y = self._update_state(y_t, current_state=Y)
#             for tau in range(self.t+1, self.T):
#                 y_tau = self.g_beta.act(Y)
#                 Y = self._update_state(y_tau, current_state=Y)
#             reward += self.process_reward(Y) / n_sample

#         return reward

    def _update_result(self, action, current_result=None):
        if current_result is None: current_result = action.reshape(1,2)
        else: current_result=np.concatenate([current_result, action.reshape(1,2)], axis= 0)
        return current_result

    def _update_state(self, action, current_state=None, tau=None):
        if current_state is None:
            if self.result is None: self.result = action.reshape(1,2)
            else: self.result=np.concatenate([self.result, action.reshape(1,2)], axis= 0)
            if (self.t+1)<self.T: self._state = self.symbols[self.t+1].reshape(1,1,2)
        else:
            if (tau+1)>=self.T: return current_state
            return self.symbols[tau+1].reshape(1,1,2)

#     def _update_state(self, state, current_state=None):
#         if current_state is None:
#             self._state[:,-1,:] = state
#             self._state = np.concatenate([self._state, self.symbols[self.t].reshape(1,1,2)], axis=1)
#         else:
#             current_state[:,-1,:] = state
#             if self.t>=self.T: return current_state
#             return np.concatenate([current_state, self.symbols[self.t].reshape(1,1,2)], axis= 1)
