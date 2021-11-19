

class AdvCAT:
    def __init__(self, scenario, name):
        self.scenario = scenario
        self.name = name
        self.advcat = scenario.Children.Item(name)
