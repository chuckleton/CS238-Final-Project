from agi.stk12.stkobjects.astrogator import (AgEVASegmentType,
    AgEVAManeuverType, AgEVAAttitudeControl, AgEVAPropulsionMethod,
    AgVAStoppingCondition)


class Satellite:
    def __init__(self, scenario, name):
        self.scenario = scenario
        self.name = name
        self.satellite = scenario.Children.Item(name)
        self.classical_elements = self.satellite.DataProviders.Item(
            'Classical Elements')

    def get_classical_elements(self, time):
        B1950_elements = self.classical_elements.Group.Item(
            'B1950').ExecSingle(time)
        element_data = B1950_elements.DataSets.GetRow(0)
        return element_data


class AstrogatorSatellite(Satellite):
    def __init__(self, scenario, name):
        super().__init__(scenario, name)
        self.driver = self.satellite.Propagator
        self.current_idx = 0
        self.reset_propagator()

    def append_impulse_by_thrust_vector(self, thrust_vector,
                                        stop_time, **kwargs):
        driver = self.driver
        maneuver = driver.MainSequence.Insert(
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

        self.insert_propagate(stop_time, before_name='Propagate')
        self.update_propagate_to_stop()
        self.current_idx += 1

    def run_mcs(self):
        self.driver.RunMCS()

    def execute_action(self, action, stop_time, **kwargs):
        self.append_impulse_by_thrust_vector(action, stop_time, **kwargs)
        self.run_mcs()

    def insert_propagate(self, TripVal, sequence_name=None, before_name='-'):
        driver = self.driver
        if sequence_name == None:
            sequence_name = f"Propagate-{self.current_idx}"
        propagate = driver.MainSequence.Insert(
            AgEVASegmentType.eVASegmentTypePropagate,
            sequence_name, before_name)
        StopDuration = propagate.StoppingConditions.Item(0)
        AgVAStoppingCondition(StopDuration.Properties).Trip = TripVal

    def update_propagate_to_stop(self):
        driver = self.driver
        propagate = driver.MainSequence.Item("Propagate")
        propToStop = propagate.StoppingConditions.Add("UserSelect")
        propagate.StoppingConditions.Remove(0)
        propToStop.Properties.Trip = self.scenario.StopTime

    def insert_propagate_to_stop(self):
        driver = self.driver
        driver.MainSequence.Insert(
            AgEVASegmentType.eVASegmentTypePropagate,
            'Propagate', '-')
        self.update_propagate_to_stop()

    def reset_propagator(self):
        sequence_names = [sequence.Name
                          for sequence in self.driver.MainSequence
                          if sequence.Name not in ['-', 'Initial State']]
        for sequence_name in sequence_names:
            self.driver.MainSequence.Remove(sequence_name)
        self.insert_propagate_to_stop()
        self.run_mcs()
        self.current_idx = 0
