import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
# from keras.layers import Input, Lambda, Activation, Dropout, Concatenate, Reshape, Add
# from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.wrappers import TimeDistributed
from scipy.stats import multivariate_normal
import tensorflow as tf
import pickle
import tensorflow_probability as tfp
from keras.models import model_from_json 

def GeneratorPretraining(H):
    def sampling_alt(args):
        mean_mu, log_var = args
        normal_dist = tfp.distributions.MultivariateNormalDiag(mean_mu, tf.exp(log_var/2))
        return normal_dist.sample()
    
    input=Input(shape=(None,2), dtype='float32', name='Input')
    out = LSTM(H, activation='relu', name='LSTM')(input)
    denseMean = Dense(2, name='DenseMean')(out)
    denseVariance = Dense(2, name='DenseVariance')(out)
    
    sampled_output = Lambda(sampling_alt, name='sampling_layer')([denseMean, denseVariance])
    
    generator_pretraining = Model(input, sampled_output)
    return generator_pretraining

class Generator():
    def __init__(self, sess, H, lr=1e-3, epsilon=0.1, beta=0.01):
        self.sess = sess
        self.H = H
        self._lr = lr
        self._epsilon = epsilon
        self._beta = beta
        self.build_graph()
        
    def set_lr(self, lr):
        self._lr = lr
    
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
        
    def set_beta(self, beta):
        self._beta = beta
    
    def build_graph(self):
        state_in = tf.placeholder(tf.float32, shape=(None,None, 2))
        action = tf.placeholder(tf.float32, shape=(None, 2))
        reward  =tf.placeholder(tf.float32, shape=(None, 1))
        
        lr = tf.placeholder(tf.float32) #default 1e-3
        epsilon = tf.placeholder(tf.float32) #default 0.1
        beta = tf.placeholder(tf.float32) #default 10000
        
        self.layers = []

        lstm = LSTM(self.H, name='LSTM', activation='relu',input_shape=(None,2)) #H=100
        out = lstm(state_in)
        self.layers.append(lstm)
        
        denseMean = Dense(2, name='DenseMean')
        denseVariance = Dense(2, name='DenseVariance')
        mean = denseMean(out)
        log_variance = denseVariance(out)
        self.layers.append(denseMean)
        self.layers.append(denseVariance) 
        
        self.normal_dist = tfp.distributions.MultivariateNormalDiag(mean, tf.exp(log_variance/2))
        
        #self.epsilon=0.05
        
        #self.sample = self.normal_dist.sample()
        
        #self.sample = tf.clip_by_norm(self.normal_dist.sample(), clip_norm=tf.norm(state_in, axis=-1)*(1+self.epsilon))
        
        
        
        x, y =self.normal_dist.sample(), state_in
        
        #x=state_in + tf.random.uniform(tf.shape(state_in), tf.math.reduce_std(state_in), tf.math.reduce_std(state_in))
        #y=state_in
        
        #self.sample=state_in + tf.random.uniform(tf.shape(state_in), tf.math.reduce_std(state_in), tf.math.reduce_std(state_in))
    
        #x=tf.reshape(x, tf.shape(y))
    
#         self.sample=tf.where(tf.norm(x,axis=-1)<=(1+self.epsilon)*tf.norm(tf.squeeze(y),axis=-1),x,tf.multiply(x,tf.reshape(tf.div(tf.norm(tf.squeeze(y),axis=-1), tf.norm(x,axis=-1)), (-1,1))))
        
        state_in_ = tf.reshape(state_in, (-1,2))
        
        self.sample = tf.clip_by_value(self.normal_dist.sample(), clip_value_min=state_in_-epsilon*tf.abs(state_in_), clip_value_max=state_in_+epsilon*tf.abs(state_in_))
        
        log_prob = tf.reshape(self.normal_dist.log_prob(action), (-1,1)) # (B, )
        loss = - tf.reduce_sum(log_prob * tf.cumsum(reward, axis=-1))
        loss -= beta * self.normal_dist.entropy()
        
        global_step = tf.Variable(0, trainable=False)
        lr_schedule = tf.train.exponential_decay(lr,decay_steps=20,global_step=global_step,decay_rate=0.95)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_schedule)
        optimizer_clipped = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
        minimize = optimizer.minimize(loss, var_list=[lstm.weights, denseMean.weights, denseVariance.weights], global_step=global_step)

        
        self.lr=lr
        self.epsilon=epsilon
        self.beta=beta
        self.log_prob = log_prob
        self.state_in = state_in
        self.action = action
        self.reward = reward
        self.mean = tf.squeeze(mean)
        self.log_variance = tf.squeeze(log_variance)
        self.prob = [mean, log_variance]
        self.minimize = minimize
        self.loss = loss
        
        self.reset_optimizer=tf.variables_initializer(optimizer.variables())
        self.init_lstm = tf.variables_initializer(lstm.weights)
        self.init_mean = tf.variables_initializer(denseMean.weights)
        self.init_var = tf.variables_initializer(denseVariance.weights)
        self.sess.run(self.init_lstm)
        self.sess.run(self.init_mean)
        self.sess.run(self.init_var)
        self.sess.run(self.reset_optimizer)

    def predict(self, state):
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in : state,
            self.epsilon: self._epsilon,
            self.lr: self._lr,
            self.beta: self._beta}
        action = self.sess.run(
            [self.sample], feed_dict)
        #print(action)
        return np.reshape(action, (-1,2))

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        #print(state.shape, action.shape, reward.shape)
        feed_dict = {
            self.state_in : state.reshape(-1,1,2),
            self.action : action.reshape(-1,2),
            self.reward : reward.reshape(-1,1),
            self.epsilon: self._epsilon,
            self.lr: self._lr,
            self.beta: self._beta}
        _, loss = self.sess.run(
            [self.minimize, self.loss],
            feed_dict)
        return loss

#     def sampling(self, mean_mu, log_var):
#         return np.random.multivariate_normal(mean=mean_mu.squeeze(), cov=np.diag(np.exp(log_var.squeeze())))

    def sampling_sequence_alt(self, T, orig_symbols):
        actions = self.predict(orig_symbols.reshape(-1,1,2))
        return actions
    
#     def sampling_sequence_alt(self, T, orig_symbols):
#         actions = None
#         for i in range(T):
#             action = self.predict(orig_symbols[i].reshape(1,1,2))
#             #print(mean, log_variance)
#             if actions is None: actions = action.reshape(-1,2)
#             else: actions = np.concatenate([actions, action.reshape(-1,2)], axis=0)
#         return actions
    
    def sampling_sequence(self, T, orig_symbols):
        curr_state = None
        actions = None
        for i in range(T):
            if curr_state is None: curr_state = orig_symbols[i].reshape(1,1,2)
            else: curr_state = np.concatenate([curr_state, orig_symbols[i].reshape(1,1,2)], axis=1)
            mean, log_variance = self.predict(curr_state)
            #print(mean, log_variance)
            action = self.sampling(mean, log_variance)
            curr_state[:,i,:]=action
            if actions is None: actions = action.reshape(-1,2)
            else: actions = np.concatenate([actions, action.reshape(-1,2)], axis=-1)
        return actions

    def generate_samples(self, T, num):
        sentences=[]
        for _ in range(num // self.B + 1):
            sequence = self.sampling_sequence(T)
            sequences.append(sequences)
        return sequences

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

# def Discriminator(dropout=0.5):
#     ap = lambda x,y: x+'_'+y
#     # Decreases parameters extractors. Decreased process. Added resnets in get_mod
#     def resnet(x,w,f,name,reg=None):
#         nm = lambda x : ap(name,x)
#         x = Conv2D(w,(1,1),activation=None,padding = 'same',name = nm('conv1') , kernel_regularizer = reg)(x)
#         x_b = x
#         x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv2'))(x)
#         x = BatchNormalization(name = nm('bn'))(x)
#         x = Activation(activation='relu',name = nm('act1'))(x)
#         x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv3'))(x)
#         x = BatchNormalization(name = nm('bn2'))(x)
#         x = Add(name = nm('add'))([x,x_b])
#         x = Activation(activation='relu',name = nm('act2'))(x)
#         return x

#     inputs = Input(shape=(256,2))
#     x = Reshape((256,2,1))(inputs)
#     x = resnet(x,16,(3,2),'1',keras.regularizers.l2(0.005))
#     # # x = resnet(x,32,(3,2),'2')
#     # x = MaxPool2D((2,1))(x)
#     i=0


#     x = resnet(x,32,(3,2),'3_{}'.format(i),keras.regularizers.l2(0.005))
#     #     x = resnet(x,32,(3,2), '4_{}'.format(i),keras.regularizers.l2(0.001))
#     x = MaxPool2D((2,1))(x)
#     #x = resnet(x,64,(3,2),'5_{}'.format(i),keras.regularizers.l2(0.001))
#     #     x = resnet(x,64,(3,2),'6_{}'.format(i),keras.regularizers.l2(0.001))
#     #x = MaxPool2D((2,2))(x)
#     x = Conv2D(16,(1,1))(x)

#     x = Flatten()(x)


#     x = Dense(10, activation='relu',kernel_regularizer = keras.regularizers.l2(0.005))(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1, activation='softmax',name = "sftmx_{}".format(i))(x)


#     discriminator = Model(inputs,x)
    
#     return discriminator
            
def Discriminator(dropout=0.5, load_path=None):
    
    if load_path is not None:
        json_file = open(load_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        discriminator = model_from_json(loaded_model_json)
        return discriminator
    
    ap = lambda x,y: x+'_'+y
    # Decreases parameters extractors. Decreased process. Added resnets in get_mod
    def resnet(x,w,f,name,reg=None):
        nm = lambda x : ap(name,x)
        x = Conv2D(w,(1,1),activation=None,padding = 'same',name = nm('conv1') , kernel_regularizer = reg)(x)
        x_b = x
        x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv2'))(x)
        x = BatchNormalization(name = nm('bn'))(x)
        x = Activation(activation='relu',name = nm('act1'))(x)
        x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv3'))(x)
        x = BatchNormalization(name = nm('bn2'))(x)
        x = Add(name = nm('add'))([x,x_b])
        x = Activation(activation='relu',name = nm('act2'))(x)
        return x

    inputs = Input(shape=(256,2))
    x = Reshape((256,2,1))(inputs)
    x = resnet(x,16,(3,2),'1',keras.regularizers.l2(0.005))
    # # x = resnet(x,32,(3,2),'2')
    #x = MaxPool2D((2,1))(x)
    i=0


    x = resnet(x,32,(3,2),'3_{}'.format(i),keras.regularizers.l2(0.005))
    #     x = resnet(x,32,(3,2), '4_{}'.format(i),keras.regularizers.l2(0.001))
    x = MaxPool2D((2,1))(x)
    #x = resnet(x,64,(3,2),'5_{}'.format(i),keras.regularizers.l2(0.002))
    #     x = resnet(x,64,(3,2),'6_{}'.format(i),keras.regularizers.l2(0.001))
    #x = MaxPool2D((2,2))(x)
    x = Conv2D(16,(1,1))(x)

    x = Flatten()(x)


    x = Dense(20, activation='relu',kernel_regularizer = keras.regularizers.l2(0.005))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid',name = "sftmx_{}".format(i))(x)


    discriminator = Model(inputs,x)
    
    return discriminator

# def Discriminator(dropout=0.5):
#     ap = lambda x,y: x+'_'+y
#     # Decreases parameters extractors. Decreased process. Added resnets in get_mod
#     def resnet(x,w,f,name,reg=None):
#         nm = lambda x : ap(name,x)
#         x = Conv2D(w,(1,1),activation=None,padding = 'same',name = nm('conv1') , kernel_regularizer = reg)(x)
#         x_b = x
#         x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv2'))(x)
#         x = BatchNormalization(name = nm('bn'))(x)
#         x = Activation(activation='relu',name = nm('act1'))(x)
#         x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv3'))(x)
#         x = BatchNormalization(name = nm('bn2'))(x)
#         x = Add(name = nm('add'))([x,x_b])
#         x = Activation(activation='relu',name = nm('act2'))(x)
#         return x

#     inputs = Input(shape=(256,))
#     x = Reshape((128,2,1))(inputs)
#     x = resnet(x,16,(3,2),'1',keras.regularizers.l2(0.00))
#     # # x = resnet(x,32,(3,2),'2')
#     x = MaxPool2D((2,1))(x)
#     i=0


#     x = resnet(x,32,(3,2),'3_{}'.format(i),keras.regularizers.l2(0.001))
#     #     x = resnet(x,32,(3,2), '4_{}'.format(i),keras.regularizers.l2(0.001))
#     x = MaxPool2D((2,1))(x)
#     x = resnet(x,64,(3,2),'5_{}'.format(i),keras.regularizers.l2(0.001))
#     #     x = resnet(x,64,(3,2),'6_{}'.format(i),keras.regularizers.l2(0.001))
#     x = MaxPool2D((2,2))(x)
#     x = Conv2D(16,(1,1))(x)

#     x = Flatten()(x)


#     x = Dense(80, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001))(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1, activation='sigmoid',name = "sftmx_{}".format(i))(x)


#     discriminator = Model(inputs,x)
    
#     return discriminator
