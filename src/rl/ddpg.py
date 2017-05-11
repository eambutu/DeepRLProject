"""
Implementation of DDPG - Deep Deterministic Policy Gradient

http://arxiv.org/pdf/1509.02971v2.pdf

Referenced the implementation from: https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
"""
import tensorflow as tf
import numpy as np
import tflearn
import gym

from rl.core import ReplayMemory, Sample
from rl.utils import process_samples

# Hyperparameters
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.9
# Soft target update
TAU = 0.01
MINIBATCH_SIZE = 64


class ActorNetwork(object):
    """
    Deterministic policy gradient.
    Input to the network is the state, output is the action under deterministic policy
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()

        # Target network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                for i in range(len(self.target_network_params))]

        # Gradient provided by critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        grads = zip(self.actor_gradients, self.network_params)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        layer1 = tflearn.fully_connected(inputs, 400, activation='relu')
        layer2 = tflearn.fully_connected(layer1, 300, activation='relu')
        # Final layer weights initalized [-3e-3, 3e-3] because tanh
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            layer2, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Nothing complicated. Outputs Q(s, a)
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.num_actor_vars = num_actor_vars

        self.state, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_state, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                for i in range(len(self.target_network_params))]

        self.td_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.td_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        state = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        layer1 = tflearn.fully_connected(state, 400, activation='relu')

        temp1 = tflearn.fully_connected(layer1, 300)
        temp2 = tflearn.fully_connected(action, 300)

        layer2 = tflearn.activation(
            tf.matmul(layer1, temp1.W) + tf.matmul(action, temp2.W) + temp2.b, activation='relu')

        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(layer2, 1, weights_init=w_init)
        return state, action, out

    def train(self, state, action, td_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.td_value: td_value
        })

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })

    def predict_target(self, state, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: state,
            self.target_action: action
        })

    def action_gradients(self, state, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

def main():
    with tf.Session() as sess:
        env = gym.make('Pendulum-v0')

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())

        # Initialize our Tensorflow variables
        sess.run(tf.initialize_all_variables())

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

        replay_buffer = ReplayMemory(10000)

        for i in range(50000):
            s = env.reset()

            ep_reward = 0
            terminal = False

            while not terminal:
                # Added exploration noise. In the literature, they use
                # the Ornstein-Uhlenbeck stochastic process for control tasks
                # that deal with momentum
                a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))

                s2, r, terminal, info = env.step(a[0])
                ep_reward += r

                cur_sample = Sample(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                    np.reshape(s2, (actor.s_dim,)), terminal)
                replay_buffer.append(cur_sample)

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.len() > MINIBATCH_SIZE:
                    samples = replay_buffer.sample(MINIBATCH_SIZE)
                    s_batch, a_batch, s2_batch, r_batch, t_batch = process_samples(samples, -1, False) 

                    # Calculate targets
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                if terminal:
                    print('Reward at episode ', i)
                    print(ep_reward)


if __name__ == '__main__':
    main()
