#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.02'


import enum

import numpy as np
from copy import deepcopy

from rlglue import BaseAgent


def quickDot(x1, x2):
    """
    Given matrices x1 and x2, return the multiplication of them
    """

    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
    return result


def oneHot(state, numStates):
    one_hot_vector = np.zeros((1, numStates))
    one_hot_vector[0, state] = 1
    return one_hot_vector


class Optimizer(enum.Enum):
    ADAM = 'adam'
    SGD = "sgd"


class InitialMethod(enum.Enum):
    NORMAL = "normal"
    SAXE = "saxe"


class ActivateFunction(enum.Enum):
    RELU = 'relu'
    SOFTMAX = 'softmax'


def ReLU(x):
    return np.maximum(x, 0)


class ActionValueNetwork:
    def __init__(self, networkConfig, optimizer):
        self.rand_generator = np.random.RandomState(networkConfig.get("seed"))
        self.num_states = networkConfig.get("num_states")
        self.num_hidden_units = networkConfig.get("num_hidden_units")
        self.num_actions = networkConfig.get("num_actions")
        self.layer_sizes = (self.num_states, self.num_hidden_units, 1)
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        self.init_method = InitialMethod(networkConfig.get("init_method"))
        self.optimizer = optimizer
        for i in range(0, len(self.layer_sizes) - 1):
            if self.init_method == InitialMethod.SAXE:
                self.weights[i]['W'] = self.initSaxe(self.layer_sizes[i], self.layer_sizes[i + 1])
                self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))
            else:
                self.weights[i]['W'] = self.initNormal(self.layer_sizes[i], self.layer_sizes[i], self.layer_sizes[i + 1])
                self.weights[i]['b'] = self.initNormal(self.layer_sizes[i], 1, self.layer_sizes[i + 1])

        self.hidden_activate = ActivateFunction(networkConfig.get('hidden_activate_function'))
        if self.hidden_activate == ActivateFunction.RELU:
            self.hidden_activate_function = ReLU
        else:
            raise ValueError("Invalid hidden activate function!")

    def getActionValues(self, s):
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = quickDot(s, W0) + b0
        x = self.hidden_activate_function(psi)

        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = quickDot(x, W1) + b1
        return q_vals

    def getGradient(self, s):
        # grads = [dict() for i in range(len(self.weights))]
        #
        # ### START CODE HERE ###
        # grads[1]["b"] = np.ones(self.weights[1]["b"].shape)
        # x = quickDot(s, self.weights[0]['W']) + self.weights[0]['b']
        # x = np.maximum(x, 0)
        # #     x[x < 0] = 0
        # dx = (x > 0).astype(float)
        #
        # grads[1]["W"] = x.T
        #
        # grads[0]["b"] = self.weights[1]["W"].T * dx
        # grads[0]["W"] = quickDot(s.T, grads[0]["b"])
        # ### END CODE HERE ###
        #
        # return grads

        psi_1xh = quickDot(s, self.weights[0]['W']) + self.weights[0]['b']
        x_1xh = np.maximum(psi_1xh, 0)
        dx_1xh = (psi_1xh > 0).astype(float)

        grads = [dict() for _ in range(len(self.weights))]
        grads[1]["W"] = x_1xh.T
        grads[1]["b"] = np.ones(self.weights[1]["b"].shape)
        grads[0]["b"] = self.weights[1]["W"].T * dx_1xh
        grads[0]["W"] = quickDot(s.T, grads[0]["b"])
        return grads

    def updateWeights(self, s, delta):
        grads = self.getGradient(s)
        g = [dict() for i in range(2)]
        for i in range(2):
            for param in self.weights[i].keys():
                g[i][param] = grads[i][param] * delta
        self.weights = self.optimizer.updateWeights(self.weights, g)

    def getTDUpdate(self, s, delta_mat):
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']

        psi = quickDot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        td_update = [dict() for i in range(len(self.weights))]

        v = delta_mat
        td_update[1]['W'] = quickDot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        v = quickDot(v, W1.T) * dx
        td_update[0]['W'] = quickDot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        return td_update

    def initNormal(self, numInputNodes, rows, cols):
        return self.rand_generator.normal(0, np.sqrt(2 / numInputNodes), (rows, cols))

    def initSaxe(self, rows, cols):
        """
        NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def getWeights(self):
        return deepcopy(self.weights)

    def setWeights(self, weights):
        self.weights = deepcopy(weights)


class Adam:
    def __init__(self, optimizerConfig):
        """Setup for the optimizer.

        Set parameters needed to setup the Adam algorithm.

        Assume optimizerConfig dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
        }
        """

        self.num_states = optimizerConfig.get("num_states")
        self.num_hidden_layer = optimizerConfig.get("num_hidden_layer")
        self.num_hidden_units = optimizerConfig.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizerConfig.get("step_size")
        self.beta_m = optimizerConfig.get("beta_m")
        self.beta_v = optimizerConfig.get("beta_v")
        self.epsilon = optimizerConfig.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer + 1)]
        self.v = [dict() for i in range(self.num_hidden_layer + 1)]

        for i in range(self.num_hidden_layer + 1):
            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i + 1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def updateWeights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (g[i][param] * g[i][param])

                ### compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                ### update weights
                weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


class SGD:
    def __init__(self, optimizerConfig):
        """Setup for the optimizer.

        Set parameters needed to setup the stochastic gradient descent method.

        Assume optimizerConfig dict contains:
        {
            step_size: float
        }
        """
        self.step_size = optimizerConfig.get("step_size")

    def updateWeights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                weights[i][param] += self.step_size * g[i][param]
        return weights


class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.num_actions = agent_info.get('num_actions')
        self.num_states = agent_info.get('num_states')
        self.discount = agent_info.get("discount")

        optimizer_config = {
            "num_states": self.num_states,
            "num_hidden_units": agent_info.get("num_hidden_units"),
            "num_hidden_layer": 1,
            "step_size": agent_info.get("step_size"),
            "beta_m": agent_info.get("beta_m"),
            "beta_v": agent_info.get("beta_v"),
            "epsilon": agent_info.get("epsilon"),
        }
        optimizer = Adam(optimizer_config)

        network_config = {
            "seed": agent_info.get("seed"),
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "num_hidden_units": agent_info.get("num_hidden_units"),
            "init_method": InitialMethod.NORMAL,
            "hidden_activate_function": ActivateFunction.RELU,
        }
        self.network = ActionValueNetwork(network_config, optimizer)

        self.last_state = None
        self.last_action = None

    def policy(self, observation):
        return self.policy_rand_generator.randint(self.num_actions)

    def agent_start(self, observation):
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_step(self, reward, observation):
        St_p1 = oneHot(observation, self.num_states)
        St = oneHot(self.last_state, self.num_states)
        Vt_p1 = self.network.getActionValues(St_p1)
        Vt = self.network.getActionValues(St)
        delta = reward + self.discount * Vt_p1 - Vt

        self.network.updateWeights(St, delta)
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_end(self, reward):
        St = oneHot(self.last_state, self.num_states)
        Vt = self.network.getActionValues(St)
        delta = reward - Vt
        self.network.updateWeights(St, delta)

    def agent_message(self, message):
        if message == 'values':
            states_values = np.zeros(self.num_states)
            for state in range(self.num_states):
                s = oneHot(state, self.num_states)
                states_values[state] = self.network.getActionValues(s)
            return states_values
        raise NotImplementedError
