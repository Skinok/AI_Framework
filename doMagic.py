# https://cdancette.fr/2018/01/03/reinforcement-learning-part3/

# Lancer l’entrainement
# La fonction d’entrainement est un peu plus complexe, puisqu’on va executer une première partie ou l’on va remplir en partie la mémoire. 
# Cela nous permettra de pouvoir créer des batch avec assez de données plus rapidement. Cette phase se déroule entre les lignes 22 et 34 du code ci-dessous.
# 
# La deuxième phase est l’entrainement du réseau. On lance un entrainement à chaque 100 mouvements. 
# On pourrait essayer d’en lancer plus ou moins souvent, l’apprentissage en serait surement impacté au niveau rapidité de convergence et qualité du minimum local. 
# En général, lorsqu’un algorithme converge trop vite, le minimum local sera moins bon.
import retro
import gym
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
from collections import deque

# Constant
SHOW_EVERY = 500

class Magician:


    def __init__(self,trainer, env):
        self.agent = trainer
        self.env = env

    def preprocess_data(self,frame):

        return 

    def stack_data(self,stacked_data, state, is_new_episode):
        # Preprocess frame
        frame = preprocess_data(state)

        print(" Glurp : " + str(frame.shape))

        if is_new_episode:
            # Clear our stacked_frames
            stacked_data = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            stacked_data.append(frame)
            stacked_data.append(frame)
            stacked_data.append(frame)
            stacked_data.append(frame)

            # Stack the frames
            stacked_state = np.stack(stacked_data, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_data.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_data, axis=2)

        return stacked_state, stacked_data

    def train(self,episodes, trainer, collecting=False, snapshot=500):
        
        batch_size = 8

        #env = retro.make(game='SpaceInvaders-Atari2600', record='.')
        env = gym.make("MountainCar-v0")

        counter = 1
        scores = []
        global_counter = 0
        losses = [0]
        epsilons = []

        # we start with a sequence to collect information, without learning
        if collecting:
            collecting_steps = 10000
            print("Collecting env without learning")
            steps = 0
            while steps < collecting_steps:
                state = env.reset()

                print("Initial state : " + state)
                done = False
                while not done:
                    steps += 1
                    action = random.randint(0, env.action_space.n - 1)  #env.get_random_action()
                    next_state, reward, done, _ = env.step(action)
                    
                    print("Next state :" + next_state)
                    print("Next state :" + next_state.shape)
                    print("Next state :" + str(next_state))

                    trainer.remember(state, action, reward, next_state, done)
                    state = next_state

        print("Starting training")

        #  An episode is a new game
        for episode in range(episodes+1):

            # Visually display this episode ? It's slow
            if episode % snapshot == 0:
                print(episode)
                render = True
            else:
                render = False

            # Fames stacked
            #stacked_frames = deque([np.zeros(frame_size) for _ in range(stack_size)], maxlen=stack_size)

            # New env : reset all variables
            state = env.reset()

            #next_state,stacked_frames = stack_frames(stacked_frames,next_state, True)
            #next_state = preprocess_data(state)
            #possible_actions = np.array(np.identity(env.action_space.n, dtype=np.int).tolist())

            score = 0
            done = False
            steps = 0

            # Game !
            print("")
            print("------")
            print("New game")

            while not done:

                if render:
                    env.render()

                steps += 1
                global_counter += 1
                #print("get_best_action state " + str(state.shape))

                action = trainer.get_best_action(state)

                # Aim is to get Epsilon around 0 when the number of episodes arrive to the end
                trainer.decay_epsilon(episode, episodes)

                #print("Epsilon = " + str(trainer.epsilon))

                next_state, reward, done, _ = env.step(action)
                
                # Replace reshape by stack_frames
                #next_state,stacked_frames = stack_frames(stacked_frames, next_state, False)
                #next_state = preprocess_data(state)

                score += reward
                trainer.remember(state, action, reward, next_state, done)  # ici on enregistre le sample dans la mémoire
                state = next_state

                #if global_counter % 0 == 0:
                if not render:
                    l = trainer.replay(batch_size)   # ici on lance le 'replay', c'est un entrainement du réseau
                    losses.append(l.history['loss'][0])

                #print(" Action : " + str(action) + " Score : " + str(score))
                if done:
                    print("Cool done ! score : " + str(score) )
                    scores.append(score)
                    epsilons.append(trainer.epsilon)

                print(" Steps : " + str(steps))
                #if steps > 200:
                #    break

            print(" Episode : " + str(episode) + " score : " + str(score))

            if episode % 200 == 0:
                print("episode: {}/{}, moves: {}, score: {}, epsilon: {}, loss: {}"
                      .format(episode, episodes, steps, score, trainer.epsilon, losses[-1]))

            #if episode > 0 and episode % snapshot == 0:
            #    trainer.save(id='iteration-%s' % episode)

        return scores, losses, epsilons
