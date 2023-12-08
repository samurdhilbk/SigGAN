from WirelessSystem.system import *

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

n_symbols = 1024
auth_addresses = ['3.40.5', '3.28.5', '3.24.5', '3.29.5', '3.33.5'] #['3.40.5', '3.28.5', '3.24.5', '3.29.5', '3.33.5', '3.32.5']
unauth_addresses = []
n_authorized = len(auth_addresses)
n_unauthorized = len(unauth_addresses)

rf_system = RFSystem(auth_addresses = auth_addresses, unauth_addresses = unauth_addresses, impersonator_address = '3.31.5', receiver_address = '3.36.5')

sig_auth_, txid_auth = rf_system.get_n_received_symbol_blocks(5000, n_symbols, authorized = 0)
sig_impersonate_, _ = rf_system.get_n_received_symbol_blocks(5000, n_symbols, authorized = 2)

sig_auth=sig_auth_
sig_impersonate=sig_impersonate_
sig_impersonate_ad=sig_impersonate

# hyper parameters
B = 64 # batch size
T = n_symbols # Max length of sentence
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

print(np.unique(txid_disc))
print(np.unique(txid_rd))

test_frac = 0.1
valid_frac  = 0.2

n_samples  = sig_rd.shape[0]

n_test = int(test_frac*n_samples)
n_valid = int(valid_frac*n_samples)
n_train = n_samples - n_test - n_valid

sig_rd_train,sig_rd_valid,sig_rd_test=split(sig_rd,n_train,n_valid,n_test)

out_train,out_valid,out_test=split(txid_disc,n_train,n_valid,n_test)

trainer.pre_train_discriminator(d_epochs=10, d_pre_sig_train=sig_rd_train, d_pre_out_train=out_train, d_pre_sig_valid=sig_rd_valid, d_pre_out_valid=out_valid, lr=1e-4)

print(trainer.test_discriminator(sig_rd_test, out_test))

print(trainer.test_discriminator(sig_impersonate_ad, np.zeros((sig_impersonate_ad.shape[0],))))
print(np.mean(trainer.predict_discriminator(sig_auth)))
print(np.mean(trainer.predict_discriminator(sig_impersonate_ad)))

#trainer.save_pre_train_d('data/', '_1_tx_8')

n_samples_im  = sig_impersonate[:20].shape[0]

n_test_im = int(test_frac*n_samples_im)
n_valid_im = int(valid_frac*n_samples_im)
n_train_im = n_samples_im - n_test_im - n_valid_im

sig_im_train,sig_im_valid,sig_im_test=split(sig_impersonate[:20],n_train_im,n_valid_im,n_test)

sig_im_train_re = sig_im_train.reshape((-1, 1, 2))
sig_im_train_re_t = sig_im_train.reshape((-1, 2))

sig_im_valid_re = sig_im_valid.reshape((-1, 1, 2))
sig_im_valid_re_t = sig_im_valid.reshape((-1, 2))

sig_im_test_re = sig_im_test.reshape((-1, 1,  2))

trainer.pre_train_generator(g_epochs=20, g_pre_data_train=[sig_im_train_re, sig_im_train_re_t], g_pre_data_valid=[sig_im_valid_re, sig_im_valid_re_t], lr=1e-3)

sig_im_test_re[0]

print(trainer.predict_pre_generator(sig_im_test_re[:1].reshape(1,1,2)))
print(trainer.predict_generator(sig_im_test_re[:1].reshape(1,1,2)))
print(trainer.predict_beta_generator(sig_im_test_re[:1].reshape(1,1,2)))
print(trainer.test_curr_discriminator_batch())
print(trainer.predict_curr_discriminator())

#trainer.reflect_pre_train() #uncomment this to reset the generator to default state
trainer.train(steps=1000)






