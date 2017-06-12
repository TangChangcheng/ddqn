import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, Input
from keras.layers import  Flatten, Dense, Activation
from keras.optimizers import Adam
from  doubleDQN import DQNAgent

plt.style.use('ggplot')

ENV_NAME = 'CartPole-v0'

model_path = 'dqn_{}.h5f'.format(ENV_NAME)

def main():
    env = gym.make(ENV_NAME)

    nb_actions = env.action_space.n

    x = Input(shape=env.observation_space.shape)
    # model = Flatten(input_shape=(1,) + env.observation_space.shape)(x)
    model = Dense(16)(x)
    model = Activation('relu')(model)
    model = Dense(16)(model)
    model = Activation('relu')(model)
    model = Dense(16)(model)
    model = Activation('relu')(model)
    model = Dense(nb_actions)(model)
    model = Activation('linear')(model)
    model = Model(inputs=x, outputs=model)

    print(model.summary())

    dqn = DQNAgent(model=model, nb_actions=nb_actions, nb_warm_up=10, target_model_update=1)

    dqn.compile(tf.train.AdamOptimizer(1e-3), metrics=['mae'])


    # dqn.load_model(model_path.format(ENV_NAME))
    loss = dqn.fit(env, nb_steps=30000, visualize=False)

    plt.semilogy(loss)
    plt.show()

    dqn.save_model(model_path)

    dqn.test(env, nb_episode=5, visualize=True)


if __name__ == '__main__':
  main()