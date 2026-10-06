import random

def sample_config(trial_id):
    return {
        "name": f"Trial {trial_id:02d}",
        "lr": 10 ** random.uniform(-3.5, -2.5),
        "dropout": random.uniform(0.25, 0.50),
        "weight_decay": 10 ** random.uniform(-5, -3),
    }


def sample_config_seg(trial_id):
    return {
        "name": f"Trial {trial_id:02d}",
        "lr": 10 ** random.uniform(-3.5, -2.5),
        "dropout": random.uniform(0.25, 0.65),
        "lstm_hidden": random.choice([16, 32, 48, 64]),
        "weight_decay": 10 ** random.uniform(-5, -3),
    }
