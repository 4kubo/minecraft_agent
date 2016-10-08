# coding:utf-8

import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Convolution3D, Convolution2D, Flatten, Dense, Merge


NUM_EPISODES = 12000  # Number of episodes the agent plays
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
SAVE_INTERVAL = 3000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time


class Agent():
    def __init__(self, num_actions, video_width, video_height, dimention):
        self.num_actions = num_actions
        self.video_width = video_width
        self.video_height = video_height
        self.dimention = dimention
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        self.repeated_action = 0

        self.life = 20
        self.food = 20

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i])
                                      for i in xrange(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        # self.saver = tf.train.Saver(q_network_weights)
        # self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        # self.summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        # if not os.path.exists(SAVE_NETWORK_PATH):
        #     os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())

        # # Load network
        # if LOAD_NETWORK:
        #     self.load_network()

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
        else:
            raise RuntimeError
        model_cnn.add(Flatten())
        model_cnn.add(Dense(512, activation='relu'))
        model_cnn.add(Dense(self.num_actions))

        if self.dimention is 3:
            s = tf.placeholder(tf.float32, [None, STATE_LENGTH, self.video_width, self.video_height, 3])
        elif self.dimention is 2:
            s = tf.placeholder(tf.float32, [None, self.video_width, self.video_height, STATE_LENGTH])
        q_values = model_cnn(s)

        return s, q_values, model_cnn


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

    def get_action(self, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, observation):
        if self.t % 100 == 0: print self.t
        if self.dimention is 3:
            observation = observation.reshape(1, self.video_width, self.video_height, 3)
            next_state = np.append(state[1:, :, :, :], observation, axis=0)
        elif self.dimention is 2:
            observation = observation.reshape(self.video_width, self.video_height, 1)
            next_state = np.append(state[:, :, 1:], observation, axis=2)

        # # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # reward = np.sign(reward)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state))
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

            # # Save network
            # if self.t % SAVE_INTERVAL == 0:
            #     save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.t))
            #     print('Successfully saved: ' + save_path)

        self.total_reward += reward
        q_v = self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
        self.total_q_max += np.max(q_v)
        self.duration += 1

        # if terminal:
        #     # Write summary
        #     if self.t >= INITIAL_REPLAY_SIZE:
        #         stats = [self.total_reward, self.total_q_max / float(self.duration),
        #                 self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
        #         for i in xrange(len(stats)):
        #             self.sess.run(self.update_ops[i], feed_dict={
        #                 self.summary_placeholders[i]: float(stats[i])
        #             })
        #         summary_str = self.sess.run(self.summary_op)
        #         self.summary_writer.add_summary(summary_str, self.episode + 1)

        #     # Debug
        #     if self.t < INITIAL_REPLAY_SIZE:
        #         mode = 'random'
        #     elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
        #         mode = 'explore'
        #     else:
        #         mode = 'exploit'
        #     print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
        #         self.episode + 1, self.t, self.duration, self.epsilon,
        #         self.total_reward, self.total_q_max / float(self.duration),
        #         self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

        #     self.total_reward = 0
        #     self.total_q_max = 0
        #     self.total_loss = 0
        #     self.duration = 0
        #     self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            # terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        # terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        # y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)
        y_batch = reward_batch + GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in xrange(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in xrange(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action
