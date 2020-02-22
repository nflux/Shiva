import torch
from shiva.helpers import networks_handler as nh
from shiva.helpers.misc import action2one_hot_v


class SupervisedNeuralNetwork(torch.nn.Module):

    def __init__(self, input_dim, output_dim, config):
        super(SupervisedNeuralNetwork, self).__init__()
        torch.manual_seed(5)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.loss = 0
        self.model = nh.DynamicLinearSequential(
                            input_dim,
                            output_dim,
                            config['layers'],
                            nh.parse_functions(torch.nn, config['activation_function']),
                            config['last_layer'],
                            getattr(torch.nn, config['output_function'])
                            if config['output_function'] is not None else None
                        ).to(self.device)
        self.optimizer = getattr(torch.optim, config['optimizer_function'])(params=self.net.parameters(),
                                                                            lr=config['learning_rate'])

        self.loss_function = getattr(torch.nn, config['loss_function'])()

    def forward(self, x):
        return self.model(x)

    def fit(self, data):
        self.optimizer.zero_grad()
        # Throwing away the rewards
        states, actions, _, actual_labels = data
        # Might want to one hot encode the actions to have the appropriate dimensions
        state_action_pairs = torch.cat([states, actions], dim=0)
        predicted_labels = self.model(state_action_pairs)
        self.loss = self.loss_function(predicted_labels, actual_labels)
        self.loss.backward()
        self.optimizer.step()

    def get_loss(self):
        return self.loss

    def __str__(self):
        return 'SupervisedNeuralNetwork'
