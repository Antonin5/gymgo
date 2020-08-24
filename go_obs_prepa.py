import numpy as np

class PrepaGoObs():
    def __init__(self, obs):
        self.obs = obs

    def new_obs(self, obs):
        self.obs = obs

    def format_state(self):
        x = np.array(self.obs[0])
        y = np.array(self.obs[1])
        state = x + y
        state = np.reshape(state, (361,))

        return state

    def format_impossible_move(self):
        inpossible_move = np.reshape(self.obs[3], (361,))

        return inpossible_move

    def current_player(self):
        current = self.obs[2][0][0]

        return int(current)
