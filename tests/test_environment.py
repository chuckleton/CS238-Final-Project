
from tf_agents.environments import utils
from cs238_final_project.collision_avoidance.environment import Environment

scenario_path = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\STK 12\\test_astrogator_collision\\test_astrogator_collision.sc"

if __name__ == "__main__":
    visible = True
    userControl = True

    environment = Environment.from_file(scenario_path, use_stk_engine=False,
                                        visible=visible, userControl=userControl)
    utils.validate_py_environment(environment, episodes=5)
