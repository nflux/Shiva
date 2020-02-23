import torch
from shiva.helpers import networks_handler as nh
from shiva.helpers.misc import action2one_hot_v


class SupervisedNeuralNetwork(torch.nn.Module):

    '''

        Eventually will need to distinguish between discrete and continuous actions.
        Need to one hot encode the actions in the discrete case which I'm currently working with.

    '''

    def __init__(self, num_features, num_labels, config):
        super(SupervisedNeuralNetwork, self).__init__()
        torch.manual_seed(5)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.loss = 0
        self.output = self.config['output']

        if self.output == 'discrete':
            self.model = SoftMaxHeadDynamicLinearNetwork(num_features, num_labels, num_labels, config)

        elif self.output == 'continuous':
            self.model = SoftMaxHeadDynamicLinearNetwork(num_features, num_labels, 0, config)

        self.optimizer = getattr(torch.optim, config['optimizer_function'])(params=self.model.parameters(),
                                                                            lr=config['learning_rate'])

        self.loss_function = getattr(torch.nn, config['loss_function'])()

    def fit(self, features, actual_labels):

        # Zero the gradient
        self.optimizer.zero_grad()

        # discrete classifications get gumble softmaxed values
        if self.output == 'discrete':
            predicted_labels = self.model(features, gumbel=True)
        # continuous classifications get softmaxed values
        elif self.output == 'continuous':
            predicted_labels = self.model(features)

        # get which label the model predicted for each set of features; I think dim = 0 will be okay
        # if something is amiss this is definitely something to check
        predicted_labels = torch.argmax(predicted_labels, dim=0)

        # get and set the loss
        self.prediction_loss = self.loss_function(predicted_labels, actual_labels)

        # update the classification model
        self.prediction_loss.backward()
        self.optimizer.step()

    def get_loss(self):
        return self.prediction_loss

    def __str__(self):
        return 'SupervisedNeuralNetwork'
