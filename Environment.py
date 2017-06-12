import numpy as np
import matplotlib.pyplot as plt
import gym

from gym import spaces
from copy import deepcopy



plt.style.use('ggplot')


interval = 0.05
status = np.arange(-3, 3, interval)
actions = np.arange(-6, 6, interval)

class Env(object):
    def __init__(self):
        self.action_shape = [1, ]
        self.action_space = spaces.Box(-0.1, 0.1, shape=self.action_shape)
        self.action = self.action_space.sample()
        # self.action_space = spaces.Discrete(len(actions))
        # self.action = actions[self.action_space.sample()]

        self.observation_shape = [1, ]
        self.observation_space = spaces.Box(-0.1, 0.1, shape=self.observation_shape)
        self._seed = 0

        self.reward = 0
        self.last_reward = 0

        self.max_step = 20

        self.nb_plot = 0
        self.is_train = True
        self.plt = plt
        self.plot_row = 1
        self.plot_col = 1

        self.coefs = [9.54151441, 6.8100, 2.76072571]
        self.reset()



    def foo(self, x):
        y = self.coefs[0]*np.power(x, 2) + self.coefs[1] * x + self.coefs[2]
        return y

    def reset(self, status=None):
        # print('\n\n--------------------------------------------------------------------------------')
        # self.coefs = np.random.rand(3)*10
        if status is None:
            self.status = np.random.random(1)*10
        else:
            self.status = np.array([status])
        self.init_status = deepcopy(self.status)
        self.loss = np.sum(self.foo(self.status))
        self.nb_step = 0

        if not self.is_train:
            self.i = 0
            self.nb_plot += 1
            self.fig = plt.figure(0)
            self.ax = self.fig.add_subplot(self.plot_row, self.plot_col, self.nb_plot)
            plt.ion()

        # print('init_loss = ', self.loss)
        return self.observe(self.loss)

    def seed(self, _int):
        np.random.seed(_int)

    def observe(self, loss):
        return np.concatenate([np.array(self.status)])

    def step(self, action):
        """

        :param action:
        :return:
            observation (object):
            reward (float): sum of rewards
            done (bool): whether to reset environment or not
            info (dict): for debug only
        """
        # print(self.status, action)
        self.nb_step += 1
        self.status += action
        # self.status += actions[action]

        self.action = action
        tmp = np.sum(self.foo(self.status))
        observation = self.observe(tmp)
        self.last_reward = self.reward
        self.reward = self.loss - tmp # - 0.1
        self.loss = tmp
        # done = np.abs(actions[action]) < 1e-4 or self.loss > 100 or self.nb_step >= self.max_step
        done = np.abs(action) < 1e-4 or self.loss > 100 or self.nb_step >= self.max_step
        info = {}
        return observation, self.reward, done, info

    def render(self, mode='human', close=False):
        print('\n\ninit: ', self.init_status)
        print('coefs: ', self.coefs)
        print('reward: ', self.reward)
        print('action: ', self.action)
        print('loss: ', self.loss)
        print('status: ', self.status)
        print('solution', self.solution())
        print('delta', self.status - self.solution())

        self.ax.plot([self.i, self.i + 1], [self.last_reward, self.reward], 'r')
        plt.pause(0.001)
        self.i += 1

    def solution(self):
        return -self.coefs[1]/self.coefs[0]/2.0
