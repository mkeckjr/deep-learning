import tensorflow

class DeepQNetwork(object):
    """Deep Q-Learning algorithm for reinforcement learning
    """

    def __init__(self,
                 network,
                 environment,
                 learning_rate,
                 discount_factor,
                 batch_size = 32,
                 freeze_frequency=1):
        self.q_network = network
        self.frozen_network = tensorflow.keras.models.clone_model(network)
        self.environment = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_capacity = 1000
        self.replay_buffer = []
        self.epsilon = 0.1
        self.max_episode_len = 10000
        self.freeze_frequency = freeze_frequency


    @tensorflow.function
    def q_values(self, input_batch, active_network=True):
        if active_network:
            return self.q_network(input_state)

        return self.frozen_network(input_state)


    @tensorflow.function
    def target_values(self, input_state):
        return self.frozen_network(input_state)


    def train(self, episodes, reset_replay_buffer=True):
        """Perform episodes of Q-learning
        """

        if reset_replay_buffer:
            self.replay_buffer = []

        for ep in range(episodes):
            state = self.environment.reset()
            state_history = [state]
            reward_history = []
            action_history = []

            for step in range(self.max_episode_len):
                coin_flip = numpy.random.random()
                if coin_flip < self.epsilon:
                    action = numpy.random.choice(self.q_network.output_shape[-1])
                else:
                    action = numpy.argmax(self.q_network(state).numpy())

                reward, next_state, is_terminal = self.environment.step(action)
                reward_history.append(reward)
                state_history.append(next_state)
                action_history.append(action)

                # sample a batch of data
                batch_inds = list(range(len(reward_history)))
                if len(reward_history) >= self.batch_size:
                    numpy.random.shuffle(batch_inds)
                    batch_inds = batch_inds[:self.batch_size]

                batch_image = numpy.array([state_history[index]
                                           for index in batch_inds])
                batch_next_image = numpy.array([state_history[index+1]
                                                for index in batch_inds])
                batch_reward = numpy.array([reward_history[index]
                                            for index in batch_inds])
                batch_actions = numpy.array([action_history[index]
                                             for index in batch_inds])

                # do the update
                with tensorflow.GradientTape() as tape:
                    # compute the reward value for being in state s using the Q-network
                    # that is frozen (previous iteration or earlier)
                    max_a = tensorflow.reduce_max(
                        self.q_values(next_image_batch,
                                      active_network=False), axis=[1]
                    )
                    target_values = reward_batch + self.discount_factor * max_a

                    # now compute the current Q-values of these actions
                    indices = list(zip(range(len(batch_actions)),
                                       batch_actions))
                    predicted_values = tensorflow.gather_nd(
                        self.q_value(image_batch), indices
                    )

                    # the loss is the sum squared difference
                    loss = tensor_flow.reduce_mean(
                        tensorflow.square((target_values - predicted_values))
                    )

                # before applying the gradients, check for freeze
                if step % self.freeze_frequency == 0:
                    self.freeze()

                # update the Q-function
                gradients = tape.gradient(loss,
                                          self.q_network.weights)
                self.optimizer.apply_gradients(
                    zip(self.q_network.weights, gradients)
                )
