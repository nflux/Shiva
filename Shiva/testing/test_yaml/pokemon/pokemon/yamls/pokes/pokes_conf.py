import yaml
import Config

confs = {
    'pikachu.yaml': {
        'Attacks':
        {
            'Cut': 20,
            'Thunder': 60,
            'Shock': 30,
            'Slam': 40
        },

        'Stats':
        {
            'Level': 20,
            'HP': 100,
            'Defense':30,
            'Attack':40,
            'Exp':0,
            'Status':'Normal'
        }
    },

    'squirtle.yaml': {
        'Attacks':
        {
            'Cut': 20,
            'Water Pump': 60,
            'Bubble': 30,
            'Slam': 40
        },

        'Stats':
        {
            'Level': 20,
            'HP': 100,
            'Defense':30,
            'Attack':40,
            'Exp':0,
            'Status':'Normal'
        }
    }
}

# [yaml.dump(data, open(file, 'w'), default_flow_style=False) for file,data in confs.items()]