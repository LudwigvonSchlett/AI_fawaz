import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if random.uniform(0, 1) < epsilone:
        action = random.choice(range(Q.shape[1]))
    else:
        action = np.argmax(Q[s])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01  # choose your own

    gamma = 0.8  # choose your own

    epsilon = 0.2  # choose your own

    n_epochs = 50000  # choose your own
    max_itr_per_epoch = 500  # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    print("Median reward = ", np.median(rewards))


    # plot the rewards in function of epochs

    epochs = list(range(1, n_epochs + 1))
    plt.plot(epochs, rewards, label="rewards per epoch", color="b")
    plt.xlabel("epochs")
    plt.ylabel("rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Training finished.\n")

    """

    Evaluate the q-learning algorihtm

    """

    env.close()
