#
# Call it this way : python launcher.py 1000
#
import time
from trainer import Trainer
from game import Game

import numpy as np 
import sys,os
# Lancer l’entrainement
# La fonction d’entrainement est un peu plus complexe, puisqu’on va executer une première partie ou l’on va remplir en partie la mémoire. 
# Cela nous permettra de pouvoir créer des batch avec assez de données plus rapidement. Cette phase se déroule entre les lignes 22 et 34 du code ci-dessous.
# 
# La deuxième phase est l’entrainement du réseau. On lance un entrainement à chaque 100 mouvements. 
# On pourrait essayer d’en lancer plus ou moins souvent, l’apprentissage en serait surement impacté au niveau rapidité de convergence et qualité du minimum local. 
# En général, lorsqu’un algorithme converge trop vite, le minimum local sera moins bon.

def train(episodes, trainer, wrong_action_p, alea, collecting=False, snapshot=5000):
    batch_size = 32
    game = Game(4, 4, wrong_action_p, alea=alea)
    counter = 1
    scores = []
    global_counter = 0
    losses = [0]
    epsilons = []

    # we start with a sequence to collect information, without learning
    if collecting:
        collecting_steps = 10000
        print("Collecting game without learning")
        steps = 0
        while steps < collecting_steps:
            state = game.reset()
            done = False
            while not done:
                steps += 1
                action = game.get_random_action()
                next_state, reward, done, _ = game.move(action)
                trainer.remember(state, action, reward, next_state, done)
                state = next_state

    print("Starting training")  
    global_counter = 0
    for e in range(episodes+1):
        state = game.generate_game()
        state = np.reshape(state, [1, 64])
        score = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            global_counter += 1
            action = trainer.get_best_action(state)
            trainer.decay_epsilon()
            next_state, reward, done, _ = game.move(action)
            next_state = np.reshape(next_state, [1, 64])
            score += reward
            trainer.remember(state, action, reward, next_state, done)  # ici on enregistre le sample dans la mémoire
            state = next_state
            if global_counter % 100 == 0:
                l = trainer.replay(batch_size)   # ici on lance le 'replay', c'est un entrainement du réseau
                losses.append(l.history['loss'][0])
            if done:
                scores.append(score)
                epsilons.append(trainer.epsilon)
            if steps > 200:
                break
        if e % 200 == 0:
            print("episode: {}/{}, moves: {}, score: {}, epsilon: {}, loss: {}"
                  .format(e, episodes, steps, score, trainer.epsilon, losses[-1]))
        if e > 0 and e % snapshot == 0:
            trainer.save(id='iteration-%s' % e)
    return scores, losses, epsilons


# name=model to load previous model
# load previous model : name="1611870402.1308398-iteration-8000"
myTrainer = Trainer(learning_rate=0.001, epsilon_decay=0.999995)
# First arguments is number of episodes
numberOfEpisodes = int(sys.argv[1])
scores, losses, epsilons = train(numberOfEpisodes, myTrainer, 0.1, True, snapshot=2000)

import matplotlib.pyplot as plt
score = np.array(scores)
score_c = np.convolve(score, np.full((10,), 1/10), mode="same")
plt.plot(score_c)
plt.show()