from SigGAN.models import GeneratorPretraining, Discriminator, Generator
from SigGAN.rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
sess = K.get_session()
import keras

class Trainer(object):
    def __init__(self, rf_system, B, T, n_authorized, g_H, d_dropout, g_lr=1e-3, d_lr=1e-3, n_sample=16, generate_samples=10000):
        self.rf_system = rf_system
        self.B, self.T = B, T
        self.g_H = g_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.generator_pre = GeneratorPretraining(g_H)
        self.agent = Agent(sess, B, g_H, g_lr)
        self.g_beta = Agent(sess, B, g_H, g_lr)
        self.discriminator = Discriminator(d_dropout)
        self.env = Environment(self.rf_system, self.discriminator, self.B, self.T, self.g_beta, n_sample=n_sample)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_data=None, d_pre_data=None,
        g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_data=g_pre_data, lr=g_lr)
        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_data=d_pre_data, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_data_train=None, g_pre_data_valid=None, lr=1e-3):
        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, loss='mse')
        print('Generator pre-training')
        self.generator_pre.summary()
        
        pre_train_g_filepath = 't_pre_train_g_weights'
        pre_train_g_c = [keras.callbacks.ModelCheckpoint(pre_train_g_filepath, monitor='val_loss',  save_best_only=True, save_weights_only=True)]
        self.generator_pre.fit(
            g_pre_data_train[0],
            g_pre_data_train[1],
            validation_data=(g_pre_data_valid[0], g_pre_data_valid[1]),
            callbacks=pre_train_g_c,
            epochs=g_epochs)
        
        self.generator_pre.load_weights(pre_train_g_filepath)
        self.reflect_pre_train()
    
    def predict_pre_generator(self, g_sig_test=None):
        return self.generator_pre.predict(g_sig_test)
    
    def predict_generator(self, g_sig_test=None):
        return self.agent.act(g_sig_test)
    
    def predict_beta_generator(self, g_sig_test=None):
        return self.g_beta.act(g_sig_test)
    
    def reset_generator_lr(self, lr):
        self.agent.generator.set_lr(lr)
        self.g_beta.generator.set_lr(lr)
        self.agent.generator.build_graph()
        self.g_beta.generator.build_graph()
    
    def pre_train_discriminator(self, d_epochs=1, d_pre_sig_train=None, d_pre_out_train=None, d_pre_sig_valid=None, d_pre_out_valid=None, lr=1e-3):
        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')

        pre_train_d_filepath = 't_pre_train_g_weights'
        pre_train_d_c = [keras.callbacks.ModelCheckpoint(pre_train_d_filepath, monitor='val_loss',  save_best_only=True, save_weights_only=True)]
        history=self.discriminator.fit(
            d_pre_sig_train,
            d_pre_out_train,
            validation_data=(d_pre_sig_valid, d_pre_out_valid),
            epochs=d_epochs,
            batch_size=self.B,
            callbacks=pre_train_d_c)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        self.discriminator.load_weights(pre_train_d_filepath)

    def norm(self, sig_u):
        pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr[:,None,None]
        print(sig_u.shape)
        return sig_u
    
    def get_predicted_sequence(self):
        orig=self.rf_system.get_real_pretx_symbol_block(self.T//2)
        return orig, self.agent.generator.sampling_sequence_alt(self.T, orig)
    
    def test_curr_discriminator_batch(self):
        n_samples=100
        perturbed_signals=np.zeros((n_samples, 256, 2))
        for i in range(n_samples):
            perturbed_signals[i,:]=self.rf_system.transmit_real_symbol_block(self.agent.generator.sampling_sequence_alt(self.T, self.rf_system.get_real_pretx_symbol_block(self.T//2)), impersonator=True)
        print("Disc Batch Accuracy: {:.3f}".format(self.test_discriminator(perturbed_signals, np.ones((n_samples,)))))
    
    def predict_curr_discriminator(self):
        perturbed_signal=self.rf_system.transmit_real_symbol_block(self.agent.generator.sampling_sequence_alt(self.T, self.rf_system.get_real_pretx_symbol_block(self.T//2)), impersonator=True)
        return self.predict_discriminator(np.expand_dims(perturbed_signal, 0)).squeeze()
    
    def predict_discriminator(self, d_sig_test=None):
        return self.discriminator.predict(d_sig_test)    
    
    def test_discriminator(self, d_sig_test=None, d_out_test=None):
        thresh=0.5
        pred=self.discriminator.predict(d_sig_test)
        thresholded=(pred>thresh).astype(int)
        thresholded=thresholded.squeeze()
        return ((thresholded==d_out_test).sum())/thresholded.shape[0]    
        
    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)
        
    def reflect_agent_to_beta(self):
        i = 0
        for layer in self.agent.generator.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                if(len(self.g_beta.generator.layers)>=(i+1)):
                    self.g_beta.generator.layers[i].set_weights(w)
                i += 1
        
    def reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                if(len(self.agent.generator.layers)>=(i+1)):
                    self.agent.generator.layers[i].set_weights(w)
                    self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=100, g_steps=1,
        g_weights_path='data/save/generator.pkl',
        verbose=True):
        print('Initial Disc Accuracy: {:.3f}'.format(self.predict_curr_discriminator()))
        for step in range(steps):
            # Generator training
            for _ in range(g_steps):
                symbols = self.rf_system.get_real_pretx_symbol_block(self.T//2)
                #print(symbols)
                self.env.reset(symbols)
                states=np.zeros((self.T,1,2))
                actions=np.zeros((self.T,2))
                rewards=np.zeros((self.T,1))
                for t in range(self.T):
                    state = self.env.get_state()
                    action = self.agent.act(state)
                    next_state, reward, is_episode_end, info = self.env.step(action)
                    #print('Reward: {:.3f}'.format(reward.squeeze()))
                    states[t,:]=state
                    actions[t,:]=action
                    rewards[t,:]=reward
                    #print(state,action,reward)
                    #self.agent.generator.update(state, action, reward)
#                     if is_episode_end:
#                         if verbose:
#                             print('Disc Accuracy: {:.3f}'.format(self.predict_curr_discriminator(self.rf_system)))
#                         break

                self.agent.generator.update(states, actions, rewards)
                #print('{:d}, {:d}: Average reward: {:.3f}'.format(step, 1, np.mean(rewards)))
                print('{:d}, {:d}: Disc Accuracy: {:.3f}, Average reward: {:.3f}'.format(step, 1, self.predict_curr_discriminator(), np.mean(rewards)))
            #if(not verbose): print('Disc Accuracy: {:.3f}'.format(self.predict_curr_discriminator(self.rf_system)))
            
            # Update env.g_beta to agent
            self.reflect_agent_to_beta()

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)
