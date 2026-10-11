from pathlib import Path

import numpy as np

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY


def test_memmap_parity(tmp_path: Path):
    folder1 = tmp_path / 'store'
    folder2 = tmp_path / 'store_episode'

    max_episodes = 5
    max_timesteps = 10
    fields = {'state': ('float', (4,)), 'action': ('int', ())}

    buffer_store = ReplayBuffer(folder1, max_episodes, max_timesteps, fields)
    buffer_episode = ReplayBuffer(folder2, max_episodes, max_timesteps, fields)

    # Generate dummy data
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randint(0, 5, size=(8,)).astype(np.int32)

    # Store using store() in a loop
    with buffer_store.one_episode():
        for t in range(8):
            buffer_store.store(state=states[t], action=actions[t])

    # Store using store_episode()
    buffer_episode.store_episode(state=states, action=actions)

    # Compare
    data_store = buffer_store.get_all_data()
    data_episode = buffer_episode.get_all_data()

    for key in fields:
        assert np.allclose(data_store[key], data_episode[key]), f"Mismatch in {key}"

def test_h5py_parity(tmp_path: Path):
    folder1 = tmp_path / 'h5py_store'
    folder2 = tmp_path / 'h5py_store_episode'

    max_episodes = 5
    max_timesteps = 10
    fields = {'state': ('float', (4,)), 'action': ('int', ())}

    buffer_store = ReplayBufferH5PY(folder1, max_episodes, max_timesteps, fields)
    buffer_episode = ReplayBufferH5PY(folder2, max_episodes, max_timesteps, fields)

    # Generate dummy data
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randint(0, 5, size=(8,)).astype(np.int32)

    # Store using store() in a loop
    with buffer_store.one_episode():
        for t in range(8):
            buffer_store.store(state=states[t], action=actions[t])

    # Store using store_episode()
    buffer_episode.store_episode(state=states, action=actions)

    # Compare
    data_store = buffer_store.get_all_data()
    data_episode = buffer_episode.get_all_data()

    for key in fields:
        assert np.allclose(data_store[key], data_episode[key]), f"Mismatch in {key}"

if __name__ == "__main__":
    test_memmap_parity(Path("/tmp/memmap_parity"))
    test_h5py_parity(Path("/tmp/h5py_parity"))
