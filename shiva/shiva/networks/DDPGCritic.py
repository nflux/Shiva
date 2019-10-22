class DDPGCritic(nn.Module):

    def __init__(self, obs_dim, action_dim, head_config, tail_config):
        super(DDPGCritic, self).__init__()
        self.net_head = DynamicLinearSequential(obs_dim, 
                                                400, 
                                                parse_layers(head_config['layers']), 
                                                parse_functions(head_config['activation_function']), 
                                                head_config['last_layer'], 
                                                get_attrs(head_config['output_function']))
                                                
        self.net_tail = DynamicLinearSequential(action_dim + 400, 
                                                1, 
                                                parse_layers(tail_config['layers']), 
                                                parse_functions(tail_config['activation_function']), 
                                                tail_config['last_layer'],
                                                get_attrs(tail_config['output_function']) )
    
    def forward(self, x, a):
        obs = self.net_head(x)
        return self.net_tail(torch.cat([obs, a], dim=1))