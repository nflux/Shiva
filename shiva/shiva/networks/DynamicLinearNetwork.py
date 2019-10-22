class DynamicLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers: list, activ_function: list, output_function: object=None):
        super(DynamicLinearNetwork, self).__init__()
        self.net = DynamicLinearSequential(input_dim, output_dim, layers, activ_function, output_function)
    def forward(self, x):
        return self.net(x)