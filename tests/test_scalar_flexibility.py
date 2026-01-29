import pytest
import numpy as np

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY

@pytest.mark.parametrize('buffer_class', [ReplayBuffer, ReplayBufferH5PY])
def test_scalar_shape_flexibility(tmp_path, buffer_class):
    buffer = buffer_class(
        tmp_path,
        max_episodes = 5,
        max_timesteps = 10,
        fields = dict(scalar_field = 'float', one_dim_field = ('float', 1)),
        meta_fields = dict(scalar_meta = 'float', one_dim_meta = ('float', 1))
    )

    # Episode 1: Scalar inputs for scalar fields, 1-dim inputs for 1-dim fields (standard)
    buffer.store_episode(
        scalar_field = np.random.randn(5),
        one_dim_field = np.random.randn(5, 1),
        scalar_meta = np.random.randn(),
        one_dim_meta = np.random.randn(1)
    )

    # Episode 2: Cross inputs
    # - scalar_field (expecting T) receives (T, 1) -> should squeeze
    # - one_dim_field (expecting T, 1) receives (T,) -> should unsqueeze
    # - scalar_meta (expecting ()) receives (1,) -> should squeeze
    # - one_dim_meta (expecting (1,)) receives () -> should unsqueeze

    buffer.store_episode(
        scalar_field = np.random.randn(5, 1), 
        one_dim_field = np.random.randn(5),
        scalar_meta = np.random.randn(1),
        one_dim_meta = np.random.randn()
    )

    assert buffer.num_episodes == 2
    
    data = buffer.get_all_data()
    assert data['scalar_field'].shape == (2, 5)
    assert data['one_dim_field'].shape == (2, 5, 1)
    assert data['scalar_meta'].shape == (2,)
    assert data['one_dim_meta'].shape == (2, 1)
