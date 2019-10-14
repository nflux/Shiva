import yaml

pikachu = {
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
}

squirtle = {
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

yaml.dump_all((pikachu, squirtle), , default_flow_style=False)