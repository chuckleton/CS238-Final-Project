

class AdvCAT:
    def __init__(self, scenario, name):
        self.scenario = scenario
        self.name = name
        self.advcat = scenario.Children.Item(name)
        self.events_by_min_range = self.advcat.DataProviders.Item(
            "Events by Min Range")

    def compute(self):
        self.scenario.Root.ExecuteCommand(f"ACAT */AdvCAT/{self.name}/ Compute")

    def get_collision_probability(self):
        self.compute()
        AdvCAT_results = self.events_by_min_range.ExecElements(
            0, self.scenario.StopTime, ['Collision Probability (Analytic)'])

        # If there are no DataSets, then there are no potential collisions in our bounds
        if AdvCAT_results.DataSets.Count <= 0:
            return 0.0

        # Otherwise, we have a potential collision, return the probability of it
        collision_probability = AdvCAT_results.DataSets.GetDataSetByName(
            'Collision Probability (Analytic)')
        collision_probability = collision_probability.GetValues()
        return collision_probability[0]
