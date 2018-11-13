import gym

from time import sleep
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model


class DqnAgent:

    def __init__(self, env, training):
        self.env = env
        self.training = training

        # constants
        # TODO
        self.learning_rate = 0.5

        # create neural network
        # TODO
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=state_shape[0], activation="relu"))
        self.model.add(Dense(4, activation="relu"))
        self.model.add(Dense(action_shape))
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        #TODO
        return self.env.action_space.sample()

    def remember(self, state, action, reward, new_state, done):
        #TODO
        pass

    def replay(self):
        #TODO
        pass

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

        while not done:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            agent.remember(state=state,
                           action=action,
                           reward=reward,
                           new_state=new_state,
                           done=done)

            agent.replay()

            env.render()
            if not training:
                sleep(0.02)
            state = new_state
            total_episode_reward += reward

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
    run_agent(env, training=True, number_of_episodes=100)

    # Test performance of the agent
    run_agent(env, training=False, number_of_episodes=10)

    # Demo
    # run_agent(env, training=False, number_of_episodes=10, model_name="cartpole-v1.model")


if __name__ == "__main__":
    main()
