class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DDPGActor, self).__init__()
        self.net = DynamicLinearSequential(obs_dim, action_dim, parse_layers(config['layers']), parse_functions(config['activation_function']), config['last_layer'], get_attrs(config['output_function']))
    def forward(self, x):
        return self.net(x)