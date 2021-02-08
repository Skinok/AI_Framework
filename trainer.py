# https://cdancette.fr/2018/01/03/reinforcement-learning-part3/

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import LeakyReLU

import tensorflow
import random
import time,os

from collections import deque

##
# High Parameters
##

class Trainer:
    def __init__(self, name=None, learning_rate=0.001, epsilon = 0.85, epsilon_decay=0.01, batch_size=30, memory_size=3000):

        self.gamma = 0.9
        self.epsilon = epsilon
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

        self.stack_size = 1

        self.name = name


    def create_or_load_model(self,state_shape, action_size):

        self.state_shape = state_shape
        self.action_size = action_size

        if self.name is not None:
            print(" Loading model : " + "model-" + self.name)
            if os.path.exists("model-" + self.name):
                model = tensorflow.keras.models.load_model("model-" + self.name)
            else:
                print("Creating new model")
                model = Sequential()

                # Layer d'entrée
                # On a ajouté une nouvelle couche à notre réseau (Dense 50) (pour lui donner une meilleur force de représentation des données).
                #   dimension : taille d'un seul échantillon de données (ici une image donc widht * height)
                model.add(Dense(4, input_shape=state_shape, activation='relu'))

                # Layer cachés (hidden)
                model.add(Dense(6, activation='relu'))
                model.add(Dense(6, activation='relu'))

                # Layer de sortie 
                #    taille : nombre d'actions possibles par l'IA dans le jeu
                model.add(Dense(self.action_size, activation='linear'))
                model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            print("Please provide a name for your model")

        self.model = model
        self.model.summary()

    def decay_epsilon(self, current_episode, nb_total_episodes):
        # Epsilon policy
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = nb_total_episodes // 2

        epsilon_decay_value = self.epsilon / (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

        if self.END_EPSILON_DECAYING >= current_episode >= self.START_EPSILON_DECAYING:
          self.epsilon -= epsilon_decay_value
    
    def get_best_action(self, state, rand=True):

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        
        # Predict the reward value based on the given state
        state = np.resize( state, (self.stack_size, len(state)) )
        act_values = self.model.predict(state)

        # Pick the action based on the predicted reward
        #print("Choose action based on prediction : ")
        #str("          " + str(act_values))
        action =  np.argmax(act_values[0])  
        return action

    # Nous allons remplacer la fonction train par une fonction remember 
    # Au lieu de lancer une étape de backpropagation, elle va tout simplement stocker ce que l’on vient de voir
    # dans une queue (une structure de données qui va supprimer les éléments entrés en premier).
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])


    #  il nous faut une fonction replay qui va piocher dans la mémoire, et donner ces données aux réseau de neurone.
    def replay(self, batch_size):
        #print(" Replay to get better")
        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        inputs = np.zeros((batch_size, 2))
        outputs = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):

            #print("state " + str(state.shape))
            #print("new size : " + str(np.array(state)))
            #print("next_state " + str(next_state.shape))

            state = np.resize( state, (self.stack_size, len(state)) )
            next_state = np.resize( next_state, (self.stack_size, len(next_state)) )

            #print("resized state " + str(state))
            #print("resized state shape " + str(state.shape))

            #print("resized next_state " + str(next_state))
            #print("resized next_state shape " + str(next_state.shape))

            target = self.model.predict( state, batch_size = 1 ) #[0]
            if done:
                target[0,action] = reward  
            else:
                #print("Next state : " + str(next_state))
                predictions = self.model.predict(next_state, batch_size = 1)
                
                #print(" predictions : " + str(predictions))
                max_future_q = np.amax( predictions[0] )
                #print("maximum : " + str(maximum))

                # self.gamma is DISCOUNT
                target[0,action] = reward + self.gamma * max_future_q

            inputs[i] = state[0]
            outputs[i] = target[0]

        #print("   Fit model")
        history = self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=batch_size, shuffle=False)

        #print("End Replay to get better i="+str(i))
        return history

    # Ainsi, ici, on va utiliser random.sample pour piocher un certain nombres d’éléments aléatoirement dans la mémoire. 
    # On crée alors nos entrées et sorties dans le bon format pour le réseau de neurone, similairement à la fonction train de l’article précédent. 
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