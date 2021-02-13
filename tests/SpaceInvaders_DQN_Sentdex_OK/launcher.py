#
# Call it this way :
# python launcher.py --name Angela --mode train --episodes 100 --epsilon 0.95 --epsilon_decay 0.98 --batch_size 8
# python launcher.py --name Angela --mode test --load Angela___445.00max__181.50avg___65.00min__1613079598.model

# Train a previous model
# python launcher.py --name Angela --mode train --load Angela___380.00max__178.50avg___40.00min__1613082541.model --episodes 100 --epsilon 0.95 --epsilon_decay 0.98 --batch_size 8

# tensorboard --logs_dir=D:\AI\AI_Framework\tradingBot_DQN_v1\logs
#
import sys,os
import argparse
import time
import numpy as np 

import gym

# our code
from DQNAgent import DQNAgent
from Magician import Magician

#
# Command line arguments
#
parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
# parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
parser.add_argument("-n", "--name", type=str, action='store', help="Please specify the name of your AI model (bob, louis, estelle...)", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
parser.add_argument("-e", "--episodes", type=str, action='store', help="Number of episodes to run", required=False)
parser.add_argument("-epsilon", "--epsilon", type=str, action='store', help="Epsilon (from 0.0 to 1.0)", required=False)
parser.add_argument("-ed", "--epsilon_decay", type=str, action='store', help="Epsilon Decay (from 0.0 to 1.0)", required=False)
parser.add_argument("-b", "--batch_size", type=str, action='store', help="Number of steps to train the model at each step of the game", required=False)
# parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
# parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics of network(such as average score on game)", required=False)
# parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)
args = parser.parse_args()
print(args)

#
# Create the environment (Game, Trading, whatever...)
#

environment = gym.make('SpaceInvaders-v0')
#environment = gym.make("MountainCar-v0")

#
# Create or load a trainer
#
#if args.load:
    # load here a new trainer model
#else:

# Give the trainer the size of the environment and the number of possible actions

#
# observation_space : API
#
# observation_space.low / observation_space.high / observation_space.shape
# observation_space.sample() / observation_space.contains()

obs_space_high = environment.observation_space.high
obs_space_low = environment.observation_space.low

#print(" observation_space : " + str(environment.observation_space))
#print(" High : " + str(environment.observation_space.high))
#print(" Low : " + str(environment.observation_space.low))
#print(" Shape : " + str(environment.observation_space.shape))

#
# Create the Agent
#
myDQNAgent = DQNAgent(name=args.name)

    


#
# Test the choosen model
#
if args.mode == "test":

    myDQNAgent = DQNAgent(name=args.name)

    magician = Magician(agent=myDQNAgent, env=environment)

    magician.test(model_to_load=args.load)

#
# Train the choosen model
#
if args.mode == "train":

    myDQNAgent.set_parameters(environment.observation_space.shape, environment.env.action_space.n, batch_size=int(args.batch_size), learning_rate=0.001,)

    if not args.load:
        myDQNAgent.prepare_new_model();
    else:
        myDQNAgent.load_model(args.load)
        
    #
    # Create the Magician who deals with the Trainer and the Environment
    #
    magician = Magician(agent=myDQNAgent, env=environment, epsilon=float(args.epsilon), epsilon_decay=float(args.epsilon_decay), epsilon_mini=0.001)

    # train the agent
    magician.doMagic(nb_episodes=int(args.episodes))



