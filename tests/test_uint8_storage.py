import pytest
import torch
import numpy as np
import shutil
from pathlib import Path
from memmap_replay_buffer import ReplayBuffer

try:
    import h5py
    from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

@pytest.fixture
def temp_folders():
    folder_memmap = Path('./test_pytest_replay_memmap_uint8')
    folder_h5py = Path('./test_pytest_replay_h5py_uint8')

    if folder_memmap.exists(): shutil.rmtree(folder_memmap)
    if folder_h5py.exists(): shutil.rmtree(folder_h5py)

    yield folder_memmap, folder_h5py

    if folder_memmap.exists(): shutil.rmtree(folder_memmap)
    if folder_h5py.exists(): shutil.rmtree(folder_h5py)

def test_uint8_storage(temp_folders):
    folder_mem, folder_h5 = temp_folders

    max_episodes = 2
    max_timesteps = 5

    fields = {
        'image': ('uint8', (3, 64, 64)),
        'action': 'int'
    }

    # Initialize memmap buffer
    rb_mem = ReplayBuffer(folder_mem, max_episodes, max_timesteps, fields)

    # Store some data
    mock_image_1 = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    mock_image_2 = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)

    rb_mem.store(image=mock_image_1, action=1)
    rb_mem.store(image=mock_image_2, action=2)
    rb_mem.advance_episode()

    data_mem = rb_mem.get_all_data()
    assert data_mem['image'].dtype == torch.uint8
    assert torch.all(data_mem['image'][0, 0] == mock_image_1)
    assert torch.all(data_mem['image'][0, 1] == mock_image_2)

    # Validate disk dtype
    disk_data_mem = np.load(str(folder_mem / 'image.data.npy'), mmap_mode='r')
    assert disk_data_mem.dtype == np.uint8

    # Repeat for H5PY if available
    if HAS_H5PY:
        rb_h5 = ReplayBufferH5PY(folder_h5, max_episodes, max_timesteps, fields)
        rb_h5.store(image=mock_image_1, action=1)
        rb_h5.store(image=mock_image_2, action=2)
        rb_h5.advance_episode()

        data_h5 = rb_h5.get_all_data()
        assert data_h5['image'].dtype == torch.uint8
        assert torch.all(data_h5['image'][0, 0] == mock_image_1)
        assert torch.all(data_h5['image'][0, 1] == mock_image_2)

        with h5py.File(str(folder_h5 / 'data.h5'), 'r') as f:
            assert f['data_image'].dtype == np.uint8
