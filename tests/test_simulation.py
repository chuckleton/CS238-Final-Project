from cs238_final_project.simulation.simulation import Simulation

scenario_path = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\STK 12\\test_astrogator_collision\\test_astrogator_collision.sc"

if __name__ == '__main__':
    visible = True
    userControl = True
    sim = Simulation.simulation_from_file(
        scenario_path, use_stk_engine=False,
        visible=visible, userControl=userControl)
    actions = [
               [10.0, 0.0, 0.0],
               [0.0, 10.0, 0.0],
               [10.0, 10.0, 0.0],
               [0.0, 0.0, 10.0],
              ]
    for action in actions:
        sim.execute_action(action)
