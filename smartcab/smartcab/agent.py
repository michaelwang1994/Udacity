import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import itertools
import pandas as pd

class QTable(object):
    def __init__(self, env):

        self.env = env
        self.lights = ['green', 'red']
        self.oncoming = ['left', 'forward', 'right', None]
        self.left = ['left', 'forward', 'right', None]
        self.right = ['left', 'forward', 'right', None]
        self.next_waypoint = ['left', 'forward', 'right']

        #TODO: {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        self.q_vals_empty = {}

        for state in list(itertools.product(*[self.lights, self.oncoming, self.left,self.right, self.next_waypoint])):
            self.q_vals_empty[state] = {'left': 0, 'forward': 0, 'right': 0, None: 0}
        self.q_vals = self.q_vals_empty

    def update(self, state, state1, action, reward, alpha = 0.8, gamma = 0.8):
        q_s_a = self.q_vals[state][action]
        q_s1_a1 = self.q_vals[state1].values()
        q_s_a = (1.0 - alpha) * q_s_a + alpha * (reward + gamma * max(q_s1_a1))
        self.q_vals[state][action] = q_s_a

    def next_move(self, state, epsilon):

        if random.random() < epsilon:
            action = random.choice(['left', 'forward', 'right', None])

            return action

        else:
            Qs = self.q_vals[state].copy()
            maxQ = max(Qs.values())
            good_moves = [m for m, v in Qs.items() if v == maxQ]
            action = random.choice(good_moves)

            return action

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha, gamma, epsilon):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table= QTable(env)
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.next_waypoint1 = None
        self.state = None
        self.state1 = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.state = [0, 0]
        self.q_table.q_vals = self.q_table.q_vals_empty

    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        # TODO: Select action according to {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        action = self.q_table.next_move(self.state, self.epsilon)
        # TODO: Learn policy based on state, action, reward
        reward = self.env.act(self, action)
        inputs1 = self.env.sense(self)
        self.next_waypoint1 = self.planner.next_waypoint()
        self.state1 = (inputs1['light'], inputs1['oncoming'], inputs1['left'], inputs1['right'], self.next_waypoint1)
        self.q_table.update(self.state, self.state1, action, reward, alpha = self.alpha, gamma = self.gamma)
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    alpha_range = np.arange(0.0, 1, 0.1)
    gamma_range = np.arange(0.0, 1, 0.1)
    epsilon_range = np.arange(0.0, 0.051, 0.01)

    df_heatmap = pd.DataFrame(index = alpha_range, columns = gamma_range)
    idx = pd.IndexSlice

    for alpha in alpha_range:
        for gamma in gamma_range:

            full_scores = []
            full_times = []
            full_rewards = []

            for epsilon in epsilon_range:

            # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_trials=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                full_scores.append(np.mean(e.score))
                full_times.append(np.mean(e.time_used))
                full_rewards.append(np.mean(e.reward_list))
            df_heatmap.loc[idx[alpha], idx[gamma]] = np.mean(full_scores)

    print df_heatmap
if __name__ == '__main__':
    run()
