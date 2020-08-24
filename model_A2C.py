import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):

    def __init__(self,num_actions: int):
        super().__init__()

        self.common1 = layers.Dense(128, activation='relu')
        self.common2 = layers.Dense(256, activation='relu')
        self.common3 = layers.Dense(512, activation='relu')
        self.common4 = layers.Dense(512, activation='relu')
        self.common5 = layers.Dense(258, activation='relu')
        self.common6 = layers.Dense(128, activation='relu')
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        x = self.common1(x)
        x = self.common2(x)
        x = self.common3(x)
        x = self.common4(x)
        x = self.common5(x)
        x = self.common6(x)

        return self.actor(x), self.critic(x)