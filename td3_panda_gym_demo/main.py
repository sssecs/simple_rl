# https://github.com/qgallouedec/panda-gym
import gymnasium as gym
import panda_gym
import numpy as np
from td3_torch import Agent
import torch
from utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make('PandaReach-v3', render_mode="human")

    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]


    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=obs_shape, tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'plots/' + 'LunarLanderContinuous_' + str(n_games) + '_games.png'

    best_score = -np.Infinity
    score_history = []

    device = torch.device("cuda")

    agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        curr_obs, curr_achgoal, curr_desgoal = observation.values()
        state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal), axis=None)

        done = False
        score = 0
        turn = 0
        while (not done) and (turn < 1000):
            action = agent.choose_action(state)
            observation_, reward, done, truncated, info = env.step(action)

            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal), axis=None)

            curr_obs, curr_achgoal, curr_desgoal = observation_.values()
            state_ = np.concatenate((curr_obs, curr_achgoal, curr_desgoal), axis=None)

            agent.remember(state, action, reward, state_, done)
            agent.learn()
            score += reward
            observation = observation_
            turn += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)
