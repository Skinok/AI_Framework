#
# Call it this way :
# python launcher.py --name Angela --episodes 1500 --mode train
# python launcher.py --load Angela --mode train --episodes 5000
# python launcher.py --load Angela --mode test
#
import sys,os
import argparse
import time
import numpy as np 

import gym

# our code
from trainer import Trainer
from doMagic import Magician

#
# Command line arguments
#
parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
# parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
parser.add_argument("-e", "--episodes", type=str, action='store', help="Number of episodes to run", required=True)
parser.add_argument("-n", "--name", type=str, action='store', help="Please specify the name of your AI model (bob, louis, estelle...)", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
# parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
# parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics of network(such as average score on game)", required=False)
# parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)
args = parser.parse_args()
print(args)


#
# Create the environment (Game, Trading, whatever...)
#
environment = gym.make("MountainCar-v0")

#
# Create or load a trainer
#
#if args.load:
    # load here a new trainer model
#else:
myTrainer = Trainer(name=args.name, batch_size=8, learning_rate=0.001, epsilon = 0.6, epsilon_decay=0.01)

# Give the trainer the size of the environment and the number of possible actions

#
# observation_space : API
#
# observation_space.low / observation_space.high / observation_space.shape
# observation_space.sample() / observation_space.contains()

obs_space_high = environment.observation_space.high
obs_space_low = environment.observation_space.low

print(" observation_space : " + str(environment.observation_space))
print(" High : " + str(environment.observation_space.high))
print(" Low : " + str(environment.observation_space.low))
print(" Shape : " + str(environment.observation_space.shape))

myTrainer.create_or_load_model(environment.observation_space.shape,environment.action_space.n)

#
# Create the Magician who deals with the Trainer and the Environment
#
magician = Magician(trainer= myTrainer, env=environment)

#
# Train the choosen model
#
if args.mode == "train":
    scores, losses, epsilons = magician.train(episodes=int(args.episodes), trainer=myTrainer, snapshot=10)


#
# Draw results => move this to another file
#
import matplotlib.pyplot as plt
score = np.array(scores)
score_c = np.convolve(score, np.full((10,), 1/10), mode="same")
plt.plot(score_c)
plt.show()