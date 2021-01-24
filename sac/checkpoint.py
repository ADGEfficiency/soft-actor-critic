import time

from sac import json


def save(actor, episode, rewards, paths):
    path = paths['run'] / 'checkpoints' / f'test-episode-{episode}'
    path.mkdir(exist_ok=True, parents=True)
    actor.save_weights(path / 'actor.h5')
    json.save(
        {
            'episode': int(episode),
            'avg-reward': np.mean(rewards),
            'episode-rewards': list(rewards)
            'time': time.utcnow().isoformat()
        },
        path / 'results.json'
    )


def load(run):
    checkpoints = Path(run) / 'checkpoints'
    checkpoints = [p for p in checkpoints.iterdir() if p.is_dir()]
    return checkpoints
