# coding:utf-8

import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Convolution3D, Convolution2D, Flatten, Dense, Merge


STATE_LENGTH = 2  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 500  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 1000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 200  # The frequency with which the target network is updated
ACTION_INTERVAL = 4  # The agent sees only every 4th input
TRAIN_INTERVAL = 10  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
TRAIN = True
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time


class Agent():
    def __init__(self, num_actions, num_state, video_width, video_height, dimention):
        self.num_actions = num_actions
        self.num_state = num_state
        self.video_width = video_width
        self.video_height = video_height
        self.dimention = dimention
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        self.repeated_action = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.i_image, self.i_state, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.t_image, self.t_state, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i])
                                      for i in xrange(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model_cnn = Sequential()
        if self.dimention is 3:
            model_cnn.add(Convolution3D(32, STATE_LENGTH, 8, 8,
                                    subsample=(STATE_LENGTH, 2, 2),
                                    activation='relu',
                                    border_mode='valid',
                                    input_shape=(STATE_LENGTH,
                                                 self.video_width,
                                                 self.video_height,
                                                 3)))
        elif self.dimention is 2:
            model_cnn.add(Convolution2D(32, 8, 8,
                                    subsample=(4, 4),
                                    activation='relu',
                                    border_mode='valid',
                                    input_shape=(self.video_width,
                                                 self.video_height,
                                                 STATE_LENGTH)))
            model_cnn.add(Convolution2D(64, 4, 4,
                                    subsample=(2, 2),
                                    border_mode='valid',
                                    activation='relu'))
        else:
            raise RuntimeError
        model_cnn.add(Flatten())
        model_cnn.add(Dense(512, activation='relu'))

        model_state = Sequential()
        model_state.add(Dense(self.num_state, activation='relu', input_shape=(self.num_state,)))

        model = Sequential()
        model.add(Merge([model_cnn, model_state], mode='concat'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions))

        if self.dimention is 3:
            pass
            # s = tf.placeholder(tf.float32, [None, STATE_LENGTH, self.video_width, self.video_height, 3])
        elif self.dimention is 2:
            i_image = tf.placeholder(tf.float32, [None, self.video_width, self.video_height, STATE_LENGTH])
        i_state = tf.placeholder(tf.float32, [None, self.num_state])
        q_values = model([i_image, i_state])

        return i_image, i_state, q_values, model


    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update

    def get_initial_state(self, observation):
        if self.dimention is 3:
            state = [observation for _ in xrange(STATE_LENGTH)]
            return np.stack(state, axis=0)
        elif self.dimention is 2:
            state = [observation for _ in xrange(STATE_LENGTH)]
            return np.stack(state, axis=2)
        else:
            raise RuntimeError

    def get_action(self, image, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.i_image: [np.float32(image / 255.0)],\
                                                                 self.i_state: [np.float32(state)]}))
            self.repeated_action = action

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state_context, action, reward, obsv_image, agent_state):
        if self.t % 100 == 0: print self.t
        if self.dimention is 3:
            pass
            # observation = observation.reshape(1, self.video_width, self.video_height, 3)
            # next_state_context = np.append(state_context[1:, :, :, :], observation, axis=0)
        elif self.dimention is 2:
            obsv_image = obsv_image.reshape(self.video_width, self.video_height, 1)
            next_state_context = np.append(state_context[:, :, 1:], obsv_image, axis=2)

        # # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # reward = np.sign(reward)

        # Store transition in replay memory
        self.replay_memory.append((state_context, action, reward, next_state_context, agent_state))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                print 'Train network'
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                print 'Update target network'
                self.sess.run(self.update_target_network)

        self.total_reward += reward
        q_v = self.q_values.eval(feed_dict={self.i_image: [np.float32(state_context / 255.0)],
                                            self.i_state: [np.float32(agent_state)]})
        self.total_q_max += np.max(q_v)
        self.duration += 1
        self.t += 1

        return next_state_context

    def train_network(self):
        context_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            context_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            state_batch.append(data[4])
            # terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        # terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.t_image: np.float32(np.array(next_state_batch) / 255.0),
                                                                     self.t_state: np.float32(state_batch)})
        # y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)
        y_batch = reward_batch + GAMMA * np.max(target_q_values_batch, axis=1)
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.i_image: np.float32(np.array(context_batch) / 255.0),
            self.i_state: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss
