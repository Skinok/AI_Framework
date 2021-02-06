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

# our code
from trainer import Trainer
import doMagic as magic

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
# Load a new trainer
#
myTrainer = Trainer(name=args.name,  learning_rate=0.001, epsilon_decay=0.999995)

#
# Train the choosen model
#
if args.mode == "train":
    scores, losses, epsilons = magic.train(episodes=int(args.episodes), trainer=myTrainer, wrong_action_p=0.1, alea=True, snapshot=2000)

#
# Draw results => move this to another file
#
import matplotlib.pyplot as plt
score = np.array(scores)
score_c = np.convolve(score, np.full((10,), 1/10), mode="same")
plt.plot(score_c)
plt.show()