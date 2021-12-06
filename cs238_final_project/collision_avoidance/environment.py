from numpy.random.mtrand import rand
import tensorflow as tf
import numpy as np

from cs238_final_project.simulation.simulation import Simulation

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import logging
logging.basicConfig(filename='collision_avoidance_environment.log',
                    encoding='utf-8', filemode='w', level=logging.INFO)


class Environment(py_environment.PyEnvironment):
    def __init__(self, simulation: Simulation, continuous = False):
        self.continuous = continuous
        self.simulation = simulation
        self.max_dist = 4.5
        self.num_angles = 4
        self.action_dtype = np.float32 if continuous else np.int32
        if continuous:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(2,), dtype=self.action_dtype, minimum=-1.0,
                maximum=1.0, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=self.action_dtype, minimum=0,
                maximum=self.num_angles, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, name='observation')
        self._episode_ended = False
        self.episode_number = 0
        self.episode_actions = []
        self.simulation_number = 0

    @property
    def _state(self):
        return np.array(self.simulation.get_state(), dtype=np.float32)

    @classmethod
    def from_file(cls, filename, continuous = False, **kwargs):
        """Create an environment from a file."""
        simulation = Simulation.simulation_from_file(filename, continuous = continuous, **kwargs)
        return cls(simulation)

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self._reset()

        if not self.continuous:
            action = int_to_angle(action,
                                    self.num_angles)
        else:
            action = xy_to_thrust_efficiency(action)
        self.update_simulation(action)
        self.episode_actions.append(action)

            # return ts.termination(self._state, reward = reward)
        # else:
        reward = self.simulation.get_reward() / self.max_dist
        reward -= 0.075*action[2]
        # Terminate early if we are clearly going the wrong way.
        if reward < -1.0:
            reward += -2.5 * (self.simulation.scenario.StopTime-self.simulation.current_time) / self.simulation.timestep
            reward -= 10.0
            logging.log(
                logging.INFO, f'Simulation ended with reward {reward} due to low reward')
            self._episode_ended = True
        if reward > -0.07:
            relative_velocity = self.get_relative_velocity()
            if relative_velocity < 0.015:
                reward += 1.0 / relative_velocity
                logging.log(
                    logging.INFO, f'Simulation ended with reward {reward} as docking was successful at relative velocity {relative_velocity}')
                self._episode_ended = True
            else:
                reward += 0.5 / relative_velocity
                logging.log(
                    logging.INFO, f'In range of target but relative velocity is too high ({relative_velocity})')
        if self.simulation_number % 100 == 0:
            self.log_data(reward=reward)
        self.simulation_number += 1
        if self._episode_ended:
            # Episode has terminated.
            if self.simulation.get_episode_ended():
                reward *= 3.0
                logging.log(
                    logging.INFO, f'Simulation ended with reward {reward}. Full duration')
            self.episode_number += 1
            return ts.termination(self._state, reward=reward)
        return ts.transition(self._state, reward=reward, discount=1.0)

    def get_relative_velocity(self):
        sim = self.simulation
        agent = sim.agent
        target = sim.target
        agent_velocity = agent.get_cartesian_velocity(sim.current_time)
        target_velocity = target.get_cartesian_velocity(sim.current_time)
        relative_velocity_keys = ['x', 'y']
        relative_velocity = {key: target_velocity[key] -
                             agent_velocity[key] for key in relative_velocity_keys}
        relative_velocity = np.linalg.norm(np.array(list(relative_velocity.values())))
        return relative_velocity

    def log_data(self, reward):
        sim = self.simulation
        agent = sim.agent
        target = sim.target
        agent_position = agent.get_cartesian_elements(sim.current_time)
        target_position = target.get_cartesian_elements(sim.current_time)
        agent_position_spherical = agent.get_mixed_spherical_elements(sim.current_time)
        target_position_spherical = target.get_mixed_spherical_elements(
            sim.current_time)
        relative_element_keys = ['x', 'y']
        relative_elements = {key: target_position[key] -
                            agent_position[key] for key in
                            relative_element_keys}

        relative_pos = sim.get_agent_relative_position()
        state = sim.get_state()
        reward = reward
        relative_velocity = self.get_relative_velocity()

        logging.info(
            f'time: {sim.current_time}, relative_pos: {relative_pos}, state: {state}, reward: {reward}')
        logging.info(
            f'agent_position: {agent_position}, target_position: {target_position}')
        logging.info(
            f'agent_position_s: {agent_position_spherical}, target_position_s: {target_position_spherical}')
        logging.info(f'relative_elements: {relative_elements}, relative_velocity: {relative_velocity}')
        logging.info('**********************************************************')

    def update_simulation(self, action):
        self.simulation.execute_action(action)
        self._episode_ended = self.simulation.get_episode_ended()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        logging.info('Resetting environment')
        # Every 25 episodes, reset the simulation to the target state to give the agent information about the target.
        if self.episode_number % 25 == 0:
            self.simulation.reset_simulation(randomize_agent=False)
        else:
            self.simulation.reset_simulation()
        self._episode_ended = False
        self.episode_actions = []
        return ts.restart(self._state)


def scalar_to_action_allow_multiple_directions(action_id, num_levels, thrust_multiplier=1):
    action = [0] * 3
    action[2] = action_id // num_levels**2
    action[1] = (action_id // num_levels) % num_levels
    action[0] = action_id % num_levels
    return thrust_multiplier*np.array(action, dtype=np.int32)

def scalar_to_action(action_id, num_levels, thrust_multiplier=1):
    action = [0] * 3
    thrust_level = action_id % num_levels
    thrust_direction = action_id // num_levels
    action[thrust_direction] = thrust_level
    return thrust_multiplier*np.array(action, dtype=np.int32)


def int_to_angle(action_id, num_angles):
    thrust_level = 1.0 if action_id > 0 else 0.0
    if action_id == 0:
        thrust_direction = 0.
    else:
        thrust_direction = (action_id-1) * 360. / num_angles
    return np.array([thrust_direction, thrust_level], dtype=np.float32)

def scalar_to_angle(action):
    action[0] *= 360.
    return action

def xy_to_thrust_efficiency(action):
    x = action[0]
    y = action[1]
    norm = np.linalg.norm(action)
    if norm > 0.01:
        x /= norm
        y /= norm
        thrust_efficiency = norm*0.5
    else:
        x = 1.
        y = 0.
        thrust_efficiency = 0.
    return np.array([x, y, thrust_efficiency], dtype=np.float32)
