import yaml

confs = {
    'brock.yaml': {
        'Pokemon':
        [
            'Growleth',
            'Squritle',
            'Ratatat'
        ],

        'Items':
        {
            'Potion':40
        }
    },

    'ash.yaml': {
        'Pokemon':
        [
            'Pikachu',
            'Pidgy'
        ],

        'Items':
        {
            'Antidote':'Normal'
        }
    }
}

# [yaml.dump(data, open(file, 'w'), default_flow_style=False) for file,data in confs.items()]