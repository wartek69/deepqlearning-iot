import gym

from time import sleep
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from collections import deque
import numpy as np
import random as random

class DqnAgent:

    def __init__(self, env, training):
        self.env = env
        self.training = training
        self.memory = deque(maxlen = 2000) # deque can append and delete from two sides of the list -> performance!

        # constants
        self.gamma = 0.95
        self.epsilon = 1 #exploitation vs exploration -> 1 = exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.005
        # create neural network
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=state_shape[0], activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(action_shape))
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=self.learning_rate))

    def act(self, state):

        if np.random.rand() <= self.epsilon and self.training:
            #explore
            return random.randrange(self.env.action_space.n)
        act_values = self.model.predict(state)
        # argmax returns index of max value, since our neural net has 2 outputs
        # it will be either a 0 or 1
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self):
        if len(self.memory) >= 32:
            samples = random.sample(self.memory, 32)
        else:
            samples = random.sample(self.memory, len(self.memory))
        states = []
        targets = []
        for state, action, reward, new_state, done in samples:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(new_state)[0])
            #keep the less optimal value, overwrite the better value
            future_target = self.model.predict(state)
            #action is a value between 0 and 1
            future_target[0][action] = target

            states.append(state[0])
            targets.append(future_target[0])
        # fit out of the lopo -> calculates overall gradient instead of gradient sample per sample
        #self.model.train_on_batch(np.asarray(states), np.asarray(targets))
        self.model.fit(np.asarray(states), np.asarray(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = load_model(name)


def run_agent(env, training=False, number_of_episodes=100, model_name=None):
    total_reward = 0
    agent = DqnAgent(env, training)

    if not training:
        try:
            if model_name is None:
                agent.load_model("{}.model".format(env.spec.id.lower()))
            else:
                agent.load_model(model_name)
        except:
            print("Failed to load {}".format(env.spec.id.lower()))
            return

    for episode in range(number_of_episodes):
        done = False
        total_episode_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])

        while not done:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            new_state = np.reshape(new_state, [1, 4])

            agent.remember(state=state,
                           action=action,
                           reward=reward,
                           new_state=new_state,
                           done=done)
            if not training:
                sleep(0.02)
                env.render()
            state = new_state
            total_episode_reward += reward
            #replay in the loop gives more training
            if training:
                agent.replay()

        print("Total reward for episode {} is {}".format(episode, total_episode_reward))
        total_reward += total_episode_reward


    if training:
        agent.save_model("{}.model".format(env.spec.id.lower()))
        print("Total training reward for agent after {} episodes is {}".format(number_of_episodes, total_reward))
    else:
        print("Result of {} = {}".format(env.spec.id, total_reward))


def main():
    env = gym.make("CartPole-v1")

    # Train the agent
    run_agent(env, training=True, number_of_episodes=500)

    # Test performance of the agent
    run_agent(env, training=False, number_of_episodes=10)

    # Demo
    # run_agent(env, training=False, number_of_episodes=10, model_name="cartpole-v1.model")


if __name__ == "__main__":
    main()
