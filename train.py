import numpy as np

from connectn_v2 import ConnectN
from learner import DQNAgent
from collections import deque
import numpy as np


if __name__ == "__main__":
    # parameters
    n_epochs = 1000

    # environment, agent
    env = ConnectN(3, 4, '2c')
    agent = DQNAgent(env.enable_actions, env.name, env.m, env.m)

    # variables
    win = 0
    n_wins_last_twenty = deque(maxlen=20)

    for cur_epoch, e in enumerate(range(n_epochs), start=1):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            # experience replay
            agent.experience_replay()

            # for log
            frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(state_t))
            if reward_t == 1:
                win += 1
                n_wins_last_twenty.append(1)
            elif reward_t == -1:
                n_wins_last_twenty.append(0)

        print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f} | WINFRAC(20): {:.03f}".format(
            e, n_epochs - 1, win, loss / frame, Q_max / frame, np.mean(n_wins_last_twenty)))

    # save model
    agent.save_model()
