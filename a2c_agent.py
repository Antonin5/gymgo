import numpy as np
import tensorflow as tf
from go_obs_prepa import PrepaGoObs
import random
import h5py
import time

class A2CAgent:
    def __init__(self, model, lr=7e-3, value_c=0.5, entropy_C=1e-4, gamma=0.99):
        self.value_c = value_c
        self.entropy_c = entropy_C
        self.gamma = gamma

        if model == None:
            self.model = tf.keras.models.load_model("model_gymg_go", custom_objects={"_logits_loss": _logits_loss, "_value_loss": _value_loss})
        else:
            self.model = model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss=[_logits_loss, _value_loss]
            )

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        prepa = PrepaGoObs(obs)
        while not done:
            current_player = prepa.current_player()
            try:
                if current_player == 0:
                    obs_format = prepa.format_state()
                    action_logit, value = self.model(obs_format[None, :])
                    action = self.play_possible_action(action_logit, prepa.format_impossible_move())
                    next_obs, reward, done, _ = env.step(action)
                    prepa.new_obs(next_obs)
                    ep_reward += reward
                else:
                    action = env.render(mode="human")
                    next_obs, _, _, _ = env.step(action)
                    prepa.new_obs(next_obs)
                if render:
                    env.render()
                obs = next_obs
            except Exception as e:
                print(e)
                pass
        return ep_reward

    def select_possible_action(self, logits, impossible_move):
        action_possibe = []

        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        if action < 361: 
            if impossible_move[action] == 1:    
                for i in range(361):
                    if impossible_move[i] == 0:
                        action_possibe.append(i)

                action_possibe.append(361)

                action = action_possibe[random.randint(0, len(action_possibe) - 1)]

        return action

    def play_possible_action(self, logits, impossible_move):
        action_possible = []

        logits = np.reshape(logits, (362,))

        for i in range(361):
            if impossible_move[i] == 0:
                action_possible.append(logits[i])
            else:
                action_possible.append(0)

        action_possible.append(logits[-1])
        print(len(action_possible))
        print(impossible_move)

        action = np.argmax(action_possible)

        return action


    def train(self, env, batch_sz=1000, updates=10000):
        actions = np.empty((2, batch_sz), dtype=np.int32)
        rewards = np.empty((2, batch_sz))
        dones = np.empty((2, batch_sz))
        values = np.empty((2, batch_sz))
        observations = np.empty((2, batch_sz) + (361,))

        ep_rewards = [0.0]
        next_obs = env.reset()
        prepa = PrepaGoObs(next_obs)

        for update in range(updates):
            debut = time.time()

            for step in range(batch_sz):
                nb_action_test = 0
                next_obs = prepa.format_state()
                current_player = prepa.current_player()
                invalid_move = prepa.format_impossible_move()
                observations[current_player][step] = next_obs
                action_logit, values[current_player][step] = self.model(next_obs[None, :])
                actions[current_player][step] = self.select_possible_action(action_logit, invalid_move)

                next_obs, rewards[current_player][step], dones[current_player][step], _ = env.step(actions[current_player][step])
                prepa.new_obs(next_obs)
                next_obs = prepa.format_state()

                if rewards[current_player][step] == 1:
                    ep_rewards[-1] += float(rewards[current_player][step])
                if rewards[current_player][step] == -1:
                    rewards[current_player][step] = 1
                    ep_rewards[-1] += -1.0

                if dones[current_player][step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    prepa.new_obs(next_obs)
                    next_obs = prepa.format_state()

            if update % 100 == 0:
                self.model.save("model_gymg_go")

            _, next_value = self.model(next_obs[None, :])
            next_value = np.squeeze(next_value, axis=-1)

            for i in range(2):
                returns, advs = self._returns_advantages(rewards, dones, values, next_value, i)
                acts_and_advs = np.concatenate([actions[i][:, None], advs[:, None]], axis=-1)
                losses = self.model.train_on_batch(observations[i],[acts_and_advs, returns])

                print(f'Episode {update}/{updates}, Episode_rewards {ep_rewards[-2]}, loss : {losses}, temps : {time.time() - debut}')
        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_values, current_player):
        returns = np.append(np.zeros_like(rewards[current_player]), next_values, axis=-1)

        for t in reversed(range(rewards[current_player].shape[0])):
            returns[t] = rewards[current_player][t] + self.gamma * returns[t + 1] * (1 - dones[current_player][t])
        returns = returns[:-1]

        advantages = returns - values[current_player]

        return returns, advantages


def _value_loss(returns, value):
    value_c = 0.5
    return value_c * tf.keras.losses.mean_squared_error(returns, value)

def _logits_loss(action_and_advantages, logits):
    entropy_c = 1e-4

    action, advantages = tf.split(action_and_advantages, 2, axis=-1)
    weighted_space_ce = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    actions = tf.cast(action, tf.int32)
    policy_loss = weighted_space_ce(actions, logits)

    probs = tf.nn.softmax(logits)
    entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

    return policy_loss - entropy_c * entropy_loss