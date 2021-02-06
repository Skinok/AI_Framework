# https://cdancette.fr/2018/01/03/reinforcement-learning-part3/

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import LeakyReLU

import tensorflow
import random
import time,os
from collections import deque

class Trainer:
    def __init__(self, name=None, learning_rate=0.001, epsilon_decay=0.9999, batch_size=30, memory_size=3000):
        self.state_size = 64
        self.action_size = 4
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # memory est la structure de données qui va nous servir de mémoire pour stocker nos ensembles (state, action, new_state, reward).
        #  C’est grâce à cette mémoire que l’on peut faire de l’experience replay.
        # A chaque action, on va remplir cette mémoire au lieu d’entrainer, 
        # Puis on va régulièrement piocher aléatoirement des samples dans cette mémoire
        # Pour lancer l’entrainement sur un batch de données.
        # dequeue Il s’agit d’une queue qui peut avoir une taille limitée, qui va supprimer automatiquement les éléments ajoutés les premiers lorsque la taille limite est atteinte.
        self.memory = deque(maxlen=memory_size)

        self.batch_size = batch_size
        
        self.name = name
        if name is not None:
            print(" Loading model : " + "model-" + name)
            model = tensorflow.keras.models.load_model("model-" + name)
            if model is None:
            	print("Failure : Could not load model")
        else:
            print("Creating new model")
            model = Sequential()
            # On a juste ajouté une nouvelle couche à notre réseau (Dense 50) (pour lui donner une meilleur force de représentation des données).
            model.add(Dense(50, input_dim=self.state_size, activation='relu'))
            model.add(Dense(30, activation='relu'))
            model.add(Dense(30, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        

        self.model = model

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
    
    def get_best_action(self, state, rand=True):

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        
        # Predict the reward value based on the given state
        act_values = self.model.predict(np.array(state))

        # Pick the action based on the predicted reward
        action =  np.argmax(act_values[0])  
        return action

    # Nous allons remplacer la fonction train par une fonction remember 
    # Au lieu de lancer une étape de backpropagation, elle va tout simplement stocker ce que l’on vient de voir
    # dans une queue (une structure de données qui va supprimer les éléments entrés en premier).
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])


    #  il nous faut une fonction replay qui va piocher dans la mémoire, et donner ces données aux réseau de neurone.
    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        inputs = np.zeros((batch_size, self.state_size))
        outputs = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.max(self.model.predict(next_state))

            inputs[i] = state
            outputs[i] = target

        return self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=batch_size)

    
    # Ainsi, ici, on va utiliser random.sample pour piocher un certain nombres d’éléments aléatoirement dans la mémoire. 
    # On crée alors nos entrées et sorties dans le bon format pour le réseau de neurone, 
    # similairement à la fonction train de l’article précédent. 
    # La différence est qu’ici, on crée un batch de plusieurs samples, au lieu de n’en donner qu’un 
    # (on voit que la dimension des input et output est (batch_size, state_size), alors qu’elle n’avait qu’une dimension précedemment.

    def save(self, id=None, overwrite=False):
        name = 'model'
        if self.name:
            name += '-' + self.name
        else:
            name += '-' + str(time.time())
        if id:
            name += '-' + id
        self.model.save(name, overwrite=overwrite)