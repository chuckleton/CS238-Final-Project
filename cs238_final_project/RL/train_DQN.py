import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.policies import random_tf_policy



from cs238_final_project.collision_avoidance.environment import Environment


class DQN:
    def __init__(self, scenario_path, **kwargs):
        self.num_iterations = 100  # @param {type:"integer"}

        self.initial_collect_steps = 10  # @param {type:"integer"}
        self.collect_steps_per_iteration = 1  # @param {type:"integer"}
        self.replay_buffer_max_length = 1000  # @param {type:"integer"}

        self.batch_size = 2  # @param {type:"integer"}
        self.learning_rate = 1e-3  # @param {type:"number"}
        self.log_interval = 1  # @param {type:"integer"}

        self.num_eval_episodes = 2  # @param {type:"integer"}
        self.eval_interval = 50  # @param {type:"integer"}

        self.n_step_update = 2  # @param {type:"integer"}

        self.metric = py_metrics.AverageReturnMetric()

        train_py_env = Environment.from_file(scenario_path, **kwargs)
        # eval_py_env = Environment.from_file(scenario_path)

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        # self.train_env = train_py_env
        # self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        self.fc_layer_params = (100, 50)
        self.action_tensor_spec = tensor_spec.from_spec(
            train_py_env.action_spec())
        self.num_actions = self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        self.dense_layers = [dense_layer(num_units) for num_units in self.fc_layer_params]
        self.q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        self.q_net = sequential.Sequential(self.dense_layers + [self.q_values_layer])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()
        data_spec = self.agent.collect_data_spec
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec, batch_size=self.train_env.batch_size, max_length=self.replay_buffer_max_length)

        # py_driver.PyDriver(
        #     self.train_env,
        #     py_tf_eager_policy.PyTFEagerPolicy(
        #         random_policy, use_tf_function=True),
        #     [rb_observer],
        #     max_steps=initial_collect_steps).run(train_py_env.reset())

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                        self.train_env.action_spec())

    def collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def reset_replay_buffer(self):
        self.replay_buffer = []

    def test_train(self):
        for i in range(self.initial_collect_steps):
            print(f'Collecting step {i}')
            self.collect_step(self.train_env, self.random_policy)

        # This loop is so common in RL, that we provide standard implementations of
        # these. For more details see the drivers module.

        # Dataset generates trajectories with shape [BxTx...] where
        # T = n_step_update + 1.
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size,
            num_steps=self.n_step_update + 1).prefetch(3)

        print(self.replay_buffer)

        self.iterator = iter(self.dataset)

    def train(self):

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step.
        self.agent.train_step_counter.assign(0)

        self.reset_replay_buffer()
        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(self.train_env, self.agent.policy, self.num_eval_episodes)
        returns = [avg_return]

        # Reset the environment.
        time_step = self.train_env.reset()

        # Create a driver to collect experience.
        collect_driver = py_driver.PyDriver(
            self.train_env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True),
            observers=self.observers,
            max_steps=self.collect_steps_per_iteration)

        for _ in range(self.num_iterations):

            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = compute_avg_return(self.train_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
