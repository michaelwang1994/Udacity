import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Q(state, action) = (1 - alpha(time)) * Q(state, action) + alpha(time) * (r + gamma * Q(next_state, next_action))


        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.state = [0, 0]
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.state = [0, 0]

    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        print self.env.q_table.q_vals
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        heading = self.env.agent_states[self]['heading']
        loc_x, loc_y = self.env.agent_states[self]['location']

        # TODO: Select action according to your policy
        action, new_heading = self.env.q_table.next_move(loc_y, loc_x, heading)

        # TODO: Update state
        action_okay = True
        if action == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif action == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif action == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        if not action_okay:
            action = None

        # TODO: Learn policy based on state, action, reward
        reward = self.env.act(self, action)
        self.env.q_table.update(loc_y, loc_x, new_heading, reward, alpha=0.5, gamma=0.8)

        if action:
            self.state[0] += 1
            self.state[1] += reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.05, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
