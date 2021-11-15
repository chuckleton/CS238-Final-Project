from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkengine import STKEngine

from simulation.satellite import Satellite, AstrogatorSatellite
from simulation.advcat import AdvCAT

class Simulation:
    def __init__(self, scenario, use_stk_engine, **kwargs):
        self.use_stk_engine = use_stk_engine
        self.scenario = scenario
        self.root = scenario.Root
        # Set the units to EpSec so we can use seconds from simulation start
        self.root.UnitPreferences.Item('DateFormat').SetCurrentUnit('EpSec')
        self.agent = AstrogatorSatellite(self.scenario, 'Satellite1')
        self.target = Satellite(self.scenario, 'Satellite2')
        self.advcat = AdvCAT(self.scenario, 'AdvCAT1')
        self.timestep = kwargs.get('timestep', 1.0)
        self.current_time = 0.0

    @classmethod
    def simulation_from_file(cls, file_path, use_stk_engine=True, **kwargs):
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
            stk = STKDesktop.StartApplication(visible=visible, userControl=userControl)

            # Get root object
            root = stk.Root

        # Load scenario from filepath
        root.LoadScenario(file_path)
        scenario = root.CurrentScenario
        return cls(scenario, use_stk_engine, **kwargs)

    def execute_action(self, action, **kwargs):
        self.agent.execute_action(action, self.current_time+self.timestep, **kwargs)
        self.current_time += self.timestep
