import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard

from collections import deque
import time
import random
from tqdm import tqdm
import os

import numpy as np
########################################################################
#
#    Hyper parameters
# 
########################################################################
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

########################################################################
#
#    
# 
########################################################################
random.seed(1)
np.random.seed(1)

class Magician:

    def __init__(self, agent = None, env = None, epsilon = 0.9, epsilon_decay=0.98, batch_size=16, learning_rate=0.001, epsilon_mini=0.001):

        # Agent
        self.agent = agent
        self.env = env

        # Parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_mini = epsilon_mini
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # For stats
        self.ep_rewards = [-200]

        # For more repetitive results
        
        tf.compat.v1.set_random_seed(1)

        # Memory fraction, used mostly when training multiple agents
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

    def setAgent(self, agent):
        self.agent = agent

    def setEnv(self, env):
        self.env = env

    def doMagic(self, nb_episodes):

        self.agent.set_parameters(self.env.observation_space.shape, self.env.action_space.n, self.batch_size, self.learning_rate)

        # Iterate over all episodes
        for episode in tqdm(range(1, nb_episodes + 1), ascii=True, unit='episodes'):

            # Update tensorboard step every episode
            self.agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, _ = self.env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    self.env.render()

                # Every step we update replay memory and train main network
                self.agent.update_replay_memory((current_state, action, reward, new_state, done))
                self.agent.train(done, step)

                current_state = new_state
                step += 1

            # Print
            print("Episode: " + str(episode) + " Score: " + str(episode_reward) + " Epsilon: " + str(self.epsilon))

            # Append episode reward to a list and log stats (every given number of episodes)
            self.ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.epsilon)

                # Save model, but only when min reward is greater or equal a set value
                filename = f'models/{self.agent.name}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'
                print(filename)
                self.agent.model.save(filename)

            # Decay epsilon
            if self.epsilon > self.epsilon_mini:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_mini, self.epsilon)

    def test(self, model_to_load):

        # Load the agent
        model = tf.keras.models.load_model("models/"+model_to_load)

        # Reset environment and get initial state
        current_state = self.env.reset()
        
        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            self.env.render()

            predicted_actions = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]

            action = np.argmax( predicted_actions )

            new_state, reward, done, _ = self.env.step(action)

            current_state = new_state