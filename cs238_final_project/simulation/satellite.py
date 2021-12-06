from agi.stk12.stkobjects.astrogator import (AgEVASegmentType,
    AgEVAManeuverType, AgEVAAttitudeControl, AgEVAPropulsionMethod,
    AgVAStoppingCondition, AgEOrbitStateType)
import numpy as np


class Satellite:
    def __init__(self, scenario, name):
        self.scenario = scenario
        self.name = name
        self.satellite = scenario.Children.Item(name)
        self.classical_elements = self.satellite.DataProviders.Item(
            'Classical Elements')
        self.mixed_spherical_elements = self.satellite.DataProviders.Item(
            'Mixed Spherical Elements')
        self.cartesian_elements = self.satellite.DataProviders.Item(
            'Cartesian Position')
        self.cartesian_velocity = self.satellite.DataProviders.Item(
            'Cartesian Velocity')

    def get_classical_elements(self, time, normalized=True, reference_frame='ICRF'):
        ICRF_elements = self.classical_elements.Group.Item(
            reference_frame).ExecSingle(time)
        element_normalizations = {'Semi-major Axis': 20000.,
                                  'Eccentricity': 1,
                                  'Inclination': 180.,
                                  'RAAN': 360.,
                                  'Arg of Perigee': 360.,
                                  'True Anomaly': 360.}
        element_data = {element: ICRF_elements.DataSets.GetDataSetByName(
            element).GetValues()[0] for element in element_normalizations.keys()}
        if normalized:
            element_data = {element: element_data[element] / element_normalizations[element]
                            for element in element_normalizations.keys()}
        return element_data

    def get_mixed_spherical_elements(self, time, normalized=True, reference_frame='ICRF'):
        ICRF_elements = self.mixed_spherical_elements.Group.Item(
            reference_frame).ExecSingle(time)
        element_normalizations = {'Detic Lon': 1.,
                                  'Detic Alt': 1.,
                                  'Horiz Flt Path Ang': 1.,
                                  'Velocity': 1.}
        element_data = {element: ICRF_elements.DataSets.GetDataSetByName(
            element).GetValues()[0] for element in element_normalizations.keys()}
        if normalized:
            element_data = {element: element_data[element] / element_normalizations[element]
                            for element in element_normalizations.keys()}
        return element_data

    def get_cartesian_elements(self, time, normalized=True, reference_frame='ICRF'):
        cartesian_elements = self.cartesian_elements.Group.Item(
            reference_frame).ExecSingle(time)
        element_normalizations = {'x': 1.,
                                  'y': 1.,
                                  'z': 1.}
        element_data = {element: cartesian_elements.DataSets.GetDataSetByName(
            element).GetValues()[0] for element in element_normalizations.keys()}
        if normalized:
            element_data = {element: element_data[element] / element_normalizations[element]
                            for element in element_normalizations.keys()}
        return element_data

    def get_cartesian_velocity(self, time, normalized=True, reference_frame='ICRF'):
        cartesian_velocity = self.cartesian_velocity.Group.Item(
            reference_frame).ExecSingle(time)
        element_normalizations = {'x': 1.,
                                  'y': 1.,
                                  'z': 1.,
                                  'radial': 1.,
                                  'in-track': 1.}
        element_data = {element: cartesian_velocity.DataSets.GetDataSetByName(
            element).GetValues()[0] for element in element_normalizations.keys()}
        if normalized:
            element_data = {element: element_data[element] / element_normalizations[element]
                            for element in element_normalizations.keys()}
        return element_data


    def get_state(self, time):
        classical_elements = np.array(list(self.get_classical_elements(time, normalized=True).values()))
        return classical_elements

class AstrogatorSatellite(Satellite):
    def __init__(self, scenario, name):
        super().__init__(scenario, name)
        self.driver = self.satellite.Propagator
        self.main_sequence = self.driver.MainSequence
        self.current_idx = 0
        self.reset_propagator()

    def random_initial_state(self, lon_range, alt_range, randomize_agent=True):
        if not randomize_agent:
            self.initial_lon = (lon_range[0] + lon_range[1]) / 2
            self.initial_alt = (alt_range[0] + alt_range[1]) / 2
        else:
            self.initial_lon = np.random.uniform(lon_range[0], lon_range[1])
            self.initial_alt = np.random.uniform(alt_range[0], alt_range[1])
        self.set_initial_state({'longitude': self.initial_lon,
                                'altitude': self.initial_alt})

    def set_initial_state(self, state):
        initial_state = self.main_sequence.Item("Initial State")
        element = initial_state.Element
        for key, value in state.items():
            if key == 'latitude':
                element.Latitude = value
            elif key == 'longitude':
                if value < 1e-6:
                    value = 0
                element.Longitude = value
            elif key == 'altitude':
                element.Altitude = value
            elif key == 'velocity_magnitude':
                element.VelocityMagnitude = value
            elif key == 'velocity_azimuth':
                element.VelocityAzimuth = value
            elif key == 'horizontal_flight_path_angle':
                element.HorizontalFlightPathAngle = value
            else:
                raise ValueError('Invalid initial state key: {}'.format(key))

    def execute_finite_maneuver(self, components, duration, thrust_efficiency=1.0, **kwargs):
        main_sequence = self.main_sequence
        maneuver = main_sequence.Item("Maneuver")
        finite_maneuver = maneuver.Maneuver
        finite_maneuver.ThrustEfficiency = thrust_efficiency
        finite_maneuver.AttitudeControl.ThrustVector.AssignXYZ(components[0], components[1], 0.0)
        StopDuration = finite_maneuver.Propagator.StoppingConditions.Item(0)
        AgVAStoppingCondition(StopDuration.Properties).Trip = duration
        driver = self.driver
        driver.BeginRun()
        maneuver.Run()
        driver.EndRun()

    def append_impulse_by_thrust_vector(self, thrust_vector,
                                        duration, **kwargs):
        main_sequence = self.main_sequence
        maneuver = main_sequence.Insert(
            AgEVASegmentType.eVASegmentTypeManeuver,
            f"Maneuver-{self.current_idx}",
            "Propagate")
        maneuver.SetManeuverType(AgEVAManeuverType.eVAManeuverTypeImpulsive)
        impulse = maneuver.Maneuver
        impulse.SetAttitudeControlType(
            AgEVAAttitudeControl.eVAAttitudeControlThrustVector)
        impulse.AttitudeControl.AssignCartesian(xVal=thrust_vector[0],
                                                yVal=thrust_vector[1],
                                                zVal=thrust_vector[2])
        impulse.SetPropulsionMethod(
            AgEVAPropulsionMethod.eVAPropulsionMethodEngineModel,
            "Constant Thrust and Isp")

        if 'UpdateMass' in kwargs:
            impulse.UpdateMass = kwargs['UpdateMass']
        else:
            impulse.UpdateMass = False

        self.insert_propagate(duration, before_name='Propagate')
        self.update_propagate_to_stop()
        self.current_idx += 1

    def run_mcs(self):
        self.driver.RunMCS()

    def execute_action(self, action, duration, **kwargs):
        if len(action) == 3:
            thrust_x = action[0]
            thrust_y = action[1]
            thrust_efficiency = action[2]
        action = np.array([thrust_x, thrust_y])
        self.execute_finite_maneuver(action, duration, thrust_efficiency=thrust_efficiency, **kwargs)


    def insert_propagate(self, TripVal, sequence_name=None, before_name='-'):
        main_sequence = self.main_sequence
        if sequence_name == None:
            sequence_name = f"Propagate-{self.current_idx}"
        propagate = main_sequence.Insert(
            AgEVASegmentType.eVASegmentTypePropagate,
            sequence_name, before_name)
        StopDuration = propagate.StoppingConditions.Item(0)
        AgVAStoppingCondition(StopDuration.Properties).Trip = TripVal

    def update_propagate_to_stop(self):
        main_sequence = self.main_sequence
        propagate = main_sequence.Item("Propagate")
        propToStop = propagate.StoppingConditions.Add("UserSelect")
        propagate.StoppingConditions.Remove(0)
        propToStop.Properties.Trip = self.scenario.StopTime

    def insert_propagate_to_stop(self):
        main_sequence = self.main_sequence
        main_sequence.Insert(
            AgEVASegmentType.eVASegmentTypePropagate,
            'Propagate', '-')
        self.update_propagate_to_stop()

    def reset_propagator(self):
        main_sequence = self.main_sequence
        maneuver = main_sequence.Item("Initial State")
        driver = self.driver
        driver.BeginRun()
        maneuver.Run()
        driver.EndRun()
        return
        main_sequence = self.main_sequence
        sequence_names = [sequence.Name
                          for sequence in main_sequence
                          if sequence.Name not in ['-', 'Initial State',
                                                   'Propagate_Backward', 'Backward Sequence']]
        for sequence_name in sequence_names:
            main_sequence.Remove(sequence_name)
        self.insert_propagate_to_stop()
        self.run_mcs()
        self.current_idx = 0
