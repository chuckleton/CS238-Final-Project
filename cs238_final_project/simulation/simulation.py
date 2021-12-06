from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkengine import STKEngine

from cs238_final_project.simulation.satellite import (Satellite,
                                                      AstrogatorSatellite)
from cs238_final_project.simulation.advcat import AdvCAT

import numpy as np

import logging
logging.basicConfig(filename='STK_simulation1.log',
                    encoding='utf-8', filemode='w', level=logging.INFO)

class Simulation:
    def __init__(self, stk, use_stk_engine, **kwargs):
        self.use_stk_engine = use_stk_engine
        self.stk = stk
        self.root = self.stk.Root
        self.scenario = self.root.CurrentScenario
        # Set the units to EpSec so we can use seconds from simulation start
        self.root.UnitPreferences.Item('DateFormat').SetCurrentUnit('EpSec')
        self.agent = AstrogatorSatellite(self.scenario, 'Agent')
        self.target = Satellite(self.scenario, 'Target')
        self.advcat = AdvCAT(self.scenario, 'AdvCAT1')
        self.timestep = kwargs.get('timestep', 3.5)
        self.reset_simulation()

    @classmethod
    def simulation_from_file(cls, filepath, use_stk_engine=True, **kwargs):
        if use_stk_engine:
            noGraphics = kwargs.get('noGraphics', True)
            # Launch STK Engine with NoGraphics mode selected
            print("Launching STK Engine...")
            stk = STKEngine.StartApplication(noGraphics=noGraphics)

            # Create root object
            root = stk.NewObjectRoot()
        else:
            visible = kwargs.get('visible', False)
            userControl = kwargs.get('userControl', False)

            # Launch GUI
            print("Launching STK...")
            stk = STKDesktop.StartApplication(visible=visible,
                                              userControl=userControl)
            # Get root object
            root = stk.Root

        # Load scenario from filepath
        root.LoadScenario(filepath)
        return cls(stk, use_stk_engine, **kwargs)

    def execute_action(self, action, **kwargs):
        self.agent.execute_action(action,
                                  self.timestep,
                                  **kwargs)
        self.current_time += self.timestep

    def reset_simulation(self, randomize_agent=True):
        target_initial_elements = self.target.get_mixed_spherical_elements(
            0.0, normalized=False)
        lon_range = np.array([-0.1, 0.1]) + target_initial_elements['Detic Lon']
        alt_range = np.array([-250, 250]) + target_initial_elements['Detic Alt']
        self.agent.random_initial_state(lon_range=lon_range,alt_range=alt_range, randomize_agent=randomize_agent)
        logging.info(
            f'Starting simulation with agent at {self.agent.initial_alt} km altitude, {self.agent.initial_lon} degrees longitude')
        self.agent.reset_propagator()
        self.current_time = 0.0
        self.initial_reward = self.get_reward()

    def get_state(self, time=None):
        if time is None:
            time = self.current_time
        agent_cartesian_velocity = self.agent.get_cartesian_velocity(time)
        relative_position = self.get_agent_relative_position(time=time)
        state = np.concatenate([np.array([agent_cartesian_velocity['x'],
                                          agent_cartesian_velocity['y']]),
                 np.array(list(relative_position.values()))]).astype(np.float32)
        return state


    def get_agent_relative_position(self, time=None):
        if time is None:
            time = self.current_time
        # agent_spherical_elements = self.agent.get_mixed_spherical_elements(
        #     time)
        # target_spherical_elements = self.target.get_mixed_spherical_elements(
        #     time)
        # relative_element_keys = ['Detic Lon', 'Detic Alt']
        # relative_elements = {key: target_spherical_elements[key] -
        #                      agent_spherical_elements[key] for key in
        #                      relative_element_keys}

        # # Account for the flip from positive to negative longitude
        # if abs(relative_elements['Detic Lon']) > 180:
        #     while relative_elements['Detic Lon'] > 180:
        #         relative_elements['Detic Lon'] -= 360
        #     while relative_elements['Detic Lon'] < -180:
        #         relative_elements['Detic Lon'] += 360
        # relative_elements['Detic Lon'] = relative_elements['Detic Lon'] * 250.
        agent_position = self.agent.get_cartesian_elements(self.current_time)
        target_position = self.target.get_cartesian_elements(self.current_time)
        relative_element_keys = ['x', 'y']
        relative_elements = {key: target_position[key] -
                             agent_position[key] for key in
                             relative_element_keys}
        return relative_elements

    def get_reward(self):
        agent_position = self.agent.get_cartesian_elements(self.current_time)
        target_position = self.target.get_cartesian_elements(self.current_time)
        relative_element_keys = ['x', 'y']
        relative_elements = {key: target_position[key] -
                             agent_position[key] for key in
                             relative_element_keys}
        error = np.sqrt(np.sum(np.square(list(relative_elements.values()))))
        return -error
        # collision_probability =  self.advcat.get_collision_probability()
        # return -1e8 * collision_probability

    def get_episode_ended(self):
        return self.current_time + self.timestep >= self.scenario.StopTime
