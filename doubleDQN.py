import tensorflow as tf
import numpy as np
import keras
import copy
import random
import keras.optimizers as optimizers

from queue import deque
from keras.models import Sequential, Input, Model
from keras.layers import Lambda



GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_STEP = (INITIAL_EPSILON - FINAL_EPSILON) / 1e3
MEMORY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, constraints, loss):
        updates = self.optimizer.get_updates(params, constraints, loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates

def clone_model(model):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    clone = Model.from_config(model.get_config())
    clone.set_weights(model.get_weights())
    return clone

class DQNAgent(object):
    def __init__(self, model, nb_actions, nb_warm_up=10, target_model_update=1e-2):
        self.model = model
        self.nb_actions = nb_actions
        self.nb_warm_up = nb_warm_up
        self.target_model_update = target_model_update if target_model_update < 1 else int(target_model_update)

        self.step = 0
        self.memory = deque(maxlen=MEMORY_SIZE)



    def compile(self, optimizer, metrics):
        self.model.compile(optimizer='Adam', loss='mse')
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='Adam', loss='mse')


        # if self.target_model_update < 1.:
        #     # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
        #     updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
        #     optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        y_pred = self.model.output
        self.y_true = tf.placeholder(tf.float32, shape=y_pred.shape)
        self.masks = tf.placeholder(tf.float32, shape=y_pred.shape)
        self.cost = tf.reduce_mean(tf.square((self.y_true - y_pred) * self.masks))
        self.optimizer = optimizer.minimize(self.cost)

        self.sess = tf.InteractiveSession()

        tf.global_variables_initializer().run()

        self.compiled = True


    def fit(self, env, nb_steps=10000, visualize=False):
        state = env.reset()
        loss = []
        total_reward = 0
        for step in range(nb_steps):
            self.step = step
            action = self.forward(state)
            next_state, reward, done, _ = env.step(action)
            metrics = self.backward(reward, next_state, not done)
            state = next_state

            loss.append(metrics)
            total_reward += reward
            if visualize:
                env.render()
            if done:
                state = env.reset()
                print('step: {step}, metrics: {metrics}, rewards: {reward}'.format(step=step, metrics=metrics, reward=total_reward))
                total_reward = 0
        return loss

    def test(self, env, nb_episode=10, visualize=True):
        self.step = self.nb_warm_up
        for step in range(nb_episode):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.forward(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if visualize:
                    env.render()
            print('episode: {step}, rewards: {reward}'.format(step=step, reward=total_reward))

    def forward(self, observation):
        action = self.select_action(observation)

        self.recent_observation = observation
        self.recent_action = action
        return action

    def backward(self, reward, next_state, terminal):
        self.memory.append((self.recent_observation, self.recent_action, reward, next_state, terminal))

        if len(self.memory) < BATCH_SIZE:
            return np.nan

        minibatch = random.sample(self.memory, BATCH_SIZE)

        state_batch = np.array([data[0] for data in minibatch])
        # state_batch = state_batch.reshape((BATCH_SIZE, -1,) + (state_batch.shape[-1],))
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])
        terminal_batch = np.array([data[4] for data in minibatch])

        actions = self.select_action_batch(next_state_batch)

        target_q_values = self.target_model.predict_on_batch(next_state_batch)
        assert target_q_values.shape == (BATCH_SIZE, self.nb_actions)
        q_batch = target_q_values[range(BATCH_SIZE), actions]

        y_batch = np.zeros([BATCH_SIZE, self.nb_actions], dtype=np.float32)
        masks = np.zeros([BATCH_SIZE, self.nb_actions], dtype=np.float32)
        calculated_q = reward_batch + GAMMA * q_batch * terminal_batch
        for Q, action, y, mask in zip(calculated_q, action_batch, y_batch, masks):
            y[action] = Q
            mask[action] = 1.0

        feed_dict = {
            self.y_true: y_batch,
            self.masks: masks,
            self.model.input: state_batch
        }

        self.optimizer.run(feed_dict)

        metrics = self.cost.eval(feed_dict)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.hard_update()

        return metrics

    def compute_q_value(self, state_batch):
        q_values = self.model.predict_on_batch(state_batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def hard_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update(self, alpha):
        pass

    def select_action(self, state):
        return self.select_action_batch(np.array([state]))[0]

    def select_action_batch(self, state_batch):

        if self.step < self.nb_warm_up: # or self.epsilon >= random.random():
            action = [random.randrange(self.nb_actions) for i in state_batch]
        else:
            q_value = self.compute_q_value(state_batch)
            assert q_value.shape == (state_batch.shape[0], self.nb_actions,)
            action = np.argmax(q_value, axis=-1)

        # # Anneal epsilon linearly over time
        # if self.start_step >= self.nb_warm_up and self.epsilon > FINAL_EPSILON :
        # 	self.epsilon -= self.epsilon_step
        return action

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        self.target_model = copy.copy(self.model)

    def save_model(self, file_path):
        self.model.save(file_path)
