# https://github.com/qgallouedec/panda-gym
import gymnasium as gym
import panda_gym
import numpy as np
from td3_torch import Agent
import torch
from utils import plot_learning_curve
import numpy as np
from HER import her_augmentation

if __name__ == '__main__':
    env = gym.make('PandaPush-v3', render_mode="human")

    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]


    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=obs_shape, tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=env.action_space.shape[0])
    n_games = 10000
    opt_steps =64
    filename = 'plots/' + 'LunarLanderContinuous_' + str(n_games) + '_games.png'

    best_score = -np.Infinity
    score_history = []

    device = torch.device("cuda")

    agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        curr_obs, curr_achgoal, curr_desgoal = observation.values()
        init_achgoal = curr_achgoal
        state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal), axis=None)

        done = False
        score = 0
        turn = 0

        obs_array = []
        actions_array = []
        new_obs_array = []

        while (not done) and (turn < 1000):
            action = agent.choose_action(state)
            observation_, reward, done, truncated, info = env.step(action)

            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal), axis=None)

            curr_obs_, curr_achgoal_, curr_desgoal_ = observation_.values()
            state_ = np.concatenate((curr_obs_, curr_achgoal_, curr_desgoal_), axis=None)

            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(observation_)

            
            agent.remember(state, action, reward, state_, done)
                
            
            
            
            score += reward
            observation = observation_
            turn += 1

        if np.linalg.norm(init_achgoal-curr_achgoal_) > 0.03:
            print("block moved")
            her_augmentation(agent, obs_array, actions_array, new_obs_array)
        for _ in range(opt_steps):
          agent.learn()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)
