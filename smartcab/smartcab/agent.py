import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha, gamma, epsilon):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.state = [0, 0]
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.state = [0, 0]

        self.env.q_table.q_vals = self.env.q_table.q_vals_empty
    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        heading = self.env.agent_states[self]['heading']
        location = self.env.agent_states[self]['location']

        # TODO: Select action according to
        action, new_heading = self.env.q_table.next_move(location, inputs, heading, epsilon = self.epsilon)

        # TODO: Learn policy based on state, action, reward
        reward = self.env.act(self, action)
        inputs1 = self.env.sense(self)
        self.env.q_table.update(location, inputs, inputs1, new_heading, reward, alpha = self.alpha, gamma = self.gamma)


        if action:
            self.state[0] += 1
            self.state[1] += reward
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    alpha_range = np.arange(0.4, 0.5, 0.1)
    gamma_range = np.arange(0.45, 0.6, 0.1)
    epsilon_range = np.arange(0.0, 0.011, 0.01)

    for alpha in alpha_range:
        for gamma in gamma_range:
            for epsilon in epsilon_range:

            # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_trials=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                print "alpha: %s, gamma: %s, epsilon: %s" % (alpha, gamma, epsilon)
                print np.mean(e.score)
                print np.mean(e.time_used)

if __name__ == '__main__':
    run()
