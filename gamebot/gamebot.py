import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


env = gym.make('SpaceInvaders-v0')# Creates the 'Cart Pole' environment
env.reset()


LR = 1e-3
goal_steps = 200# frames
score_requirement = 100
initial_games = 10# Was 10000



def initial_population():

    training_data = []
    scores = []
    accepted_scores = []

    for episodes in range(initial_games):

        score = 0
        game_memory = []
        prev_observation = []

        for step in range(goal_steps):

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])


            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:

                # One-Hot encoded
                if data[1] == 0:
                    output = [1,0,0,0,0,0]

                elif data[1] == 1:
                    output = [0,1,0,0,0,0]

                elif data[1] == 2:
                    output = [0,0,1,0,0,0]

                elif data[1] == 3:
                    output = [0,0,0,1,0,0]

                elif data[1] == 4:
                    output = [0,0,0,0,1,0]

                elif data[1] == 5:
                    output = [0,0,0,0,0,1]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('spaceinvaders.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return  training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)


    network = fully_connected(network, 6, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model= False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size= len(X[0]))

    model.fit({'input':X}, {'targets':y},
              n_epoch=5,# Number of EPOCHS
              snapshot_step=500,
              show_metric=False,
              run_id='Atari-SpaceInvaders'
              )

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for frame in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))


model.save('space_invaders.model')# Saves the trained model

#model.load('original_model.model')# Loads the trained model
#initial_population()

