#!/usr/bin/env python
# coding: utf-8
from WirelessSystem.system import *

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import keras

import os
import numpy as np
from SigGAN.train import Trainer

import tensorflow as tf

def norm(sig_u):
    pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
    sig_u = sig_u/pwr[:,None,None]
    print(sig_u.shape)
    return sig_u

def shuffle(vec1,vec2,seed = 0):
    np.random.seed(0)
    shfl_indx = np.arange(vec1.shape[0])
    np.random.shuffle(shfl_indx)
    shfl_indx = shfl_indx.astype('int')
    vec1 = vec1[shfl_indx]
    vec2 = np.copy(vec2[shfl_indx])
    return vec1,vec2

def split(vec,n_train,n_valid,n_test):
    vec_train = vec[0:n_train]
    vec_valid = vec[n_train:n_train+n_valid]
    vec_test = vec[n_train+n_valid:]
    return vec_train,vec_valid,vec_test



n_symbols = 256 // 2
sess_name = 'test_transmitters_snr'



for n_authorized in [5,10,15,25,50,100]:
    for snr in [0,5,10,20,50]:
        
        n_unauthorized = n_authorized
        suffix='_t_%d_snr_%d'%(n_authorized,snr)
        
        rf_system = RFSystem(n_authorized = n_authorized, n_unauthorized = n_unauthorized, snr = snr, sess_name = sess_name, suffix=suffix)

        sig_auth_, txid_auth = rf_system.get_n_received_symbol_blocks(5000, n_symbols, authorized = 0)
        sig_unauth_, txid_unauth = rf_system.get_n_received_symbol_blocks(5000, n_symbols, authorized = 1)
        sig_impersonate_, _ = rf_system.get_n_received_symbol_blocks(10, n_symbols, authorized = 2)


        sig_auth=sig_auth_
        sig_unauth=sig_unauth_
        sig_impersonate_ad_, _ = rf_system.get_n_received_symbol_blocks(5000, n_symbols, authorized = 2)
        sig_impersonate_ad=sig_impersonate_ad_
        sig_impersonate=sig_impersonate_.reshape((-1,2))

        # hyper parameters
        B = 64 # batch size
        T = 256 # Max length of sentence
        g_H = 100 # Generator LSTM hidden size
        g_lr = 1e-3
        d_dropout = 0.5 # dropout ratio
        d_lr = 1e-3

        n_sample=1 # Number of Monte Calro Search
        generate_samples = 2000 # Number of generated sentences

        # Pretraining parameters
        g_pre_lr = 1e-3
        d_pre_lr = 1e-3
        g_pre_epochs= 60
        d_pre_epochs = 1


        trainer = Trainer(rf_system, B, T, n_authorized, g_H, d_dropout, g_lr=g_lr, d_lr=d_lr, n_sample=n_sample, generate_samples=generate_samples)

        sig_rd = np.concatenate([sig_auth,sig_impersonate_ad])
        txid_rd = np.concatenate([txid_auth,np.ones((sig_impersonate_ad.shape[0],))*n_authorized])
        sig_rd, txid_rd = shuffle(sig_rd, txid_rd)

        txid_disc = txid_rd == n_authorized
        txid_disc = np.invert(txid_disc)
        txid_disc = txid_disc.astype(int)

        test_frac = 0.1
        valid_frac  = 0.2

        n_samples  = sig_rd.shape[0]

        n_test = int(test_frac*n_samples)
        n_valid = int(valid_frac*n_samples)
        n_train = n_samples - n_test - n_valid

        sig_rd_train,sig_rd_valid,sig_rd_test=split(sig_rd,n_train,n_valid,n_test)
        out_train,out_valid,out_test=split(txid_disc,n_train,n_valid,n_test)

        trainer.load_pre_train_d(rf_system.full_sess_dir, suffix=suffix)

        tr=trainer.test_discriminator(sig_rd_train, out_train)
        va=trainer.test_discriminator(sig_rd_valid, out_valid)
        te=trainer.test_discriminator(sig_rd_test, out_test)
        im=trainer.test_discriminator(sig_impersonate_ad, np.zeros((sig_impersonate_ad.shape[0],)))

        au_act=np.mean(trainer.predict_discriminator(sig_auth))
        im_act=np.mean(trainer.predict_discriminator(sig_impersonate_ad))
        
        assert(tr>=0.8 and va>=0.8 and te>=0.8 and im>=0.9 and au_act>=0.7 and im_act<=0.3)
        
        n_samples_im  = sig_impersonate.shape[0]

        n_test_im = int(test_frac*n_samples_im)
        n_valid_im = int(valid_frac*n_samples_im)
        n_train_im = n_samples_im - n_test_im - n_valid_im

        sig_im_train,sig_im_valid,sig_im_test=split(sig_impersonate,n_train_im,n_valid_im,n_test)
        sig_im_train_re = sig_im_train.reshape((-1, 1, 2))
        sig_im_train_re_t = sig_im_train.reshape((-1, 2))
        sig_im_valid_re = sig_im_valid.reshape((-1, 1, 2))
        sig_im_valid_re_t = sig_im_valid.reshape((-1, 2))
        sig_im_test_re = sig_im_test.reshape((-1, 1,  2))

        trainer.pre_train_generator(g_epochs=100, g_pre_data_train=[sig_im_train_re, sig_im_train_re_t], g_pre_data_valid=[sig_im_valid_re, sig_im_valid_re_t], lr=1e-3)

        pred_curr=trainer.predict_curr_discriminator()

        assert(pred_curr<=0.4)
        
        [0., 0.02, 0.05,  0.07, 0.08, 0.10, 0.12, 0.2]
        
        eps = [0., 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.2]
        betas = [0, 1000, 5000, 7500, 10000, 12500, 15000, 20000]
        reward_types = [True, False]
        
        for epsilon in eps:
            for beta in betas:
                for reward_type in reward_types:
                    trainer.reset_generator(lr=1e-3, epsilon=0.05, beta=5000, binary_reward=True)
                    trainer.reflect_pre_train()
                    trainer.train(steps=1000)
