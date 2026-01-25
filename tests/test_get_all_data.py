import torch
from memmap_replay_buffer import ReplayBuffer
import shutil
from pathlib import Path

def test_get_all_data_subselection():
    folder = Path('./test_get_all_data_data')
    if folder.exists():
        shutil.rmtree(folder)

    replay_buffer = ReplayBuffer(
        folder,
        max_episodes = 10,
        max_timesteps = 5,
        fields = dict(
            state = ('float', (8,)),
            action = 'int',
        ),
        meta_fields = dict(
            task_id = 'int',
        ),
        overwrite = True
    )

    with replay_buffer.one_episode(task_id = 1):
        for i in range(3):
            replay_buffer.store(
                state = torch.ones((8,)) * i,
                action = i
            )

    # 1. Test get_all_data without arguments (should return all)
    all_data = replay_buffer.get_all_data()
    assert 'state' in all_data
    assert 'action' in all_data
    assert 'task_id' in all_data
    assert 'episode_lens' in all_data
    assert all_data['state'].shape == (1, 3, 8)
    assert all_data['action'].shape == (1, 3)
    assert all_data['task_id'].shape == (1,)

    # 2. Test get_all_data with specific fields
    selected_fields = replay_buffer.get_all_data(fields = ('action',))
    assert 'action' in selected_fields
    assert 'state' not in selected_fields
    assert 'task_id' not in selected_fields # should be empty now
    assert selected_fields['action'].shape == (1, 3)

    # 3. Test get_all_data with specific meta fields
    selected_meta = replay_buffer.get_all_data(meta_fields = ('task_id',))
    assert 'state' not in selected_meta # should be empty now
    assert 'task_id' in selected_meta
    assert 'episode_lens' not in selected_meta
    assert selected_meta['task_id'].shape == (1,)

    # 4. Test get_all_data with both
    selected_both = replay_buffer.get_all_data(fields = ('action',), meta_fields = ('task_id',))
    assert 'state' not in selected_both
    assert 'action' in selected_both
    assert 'task_id' in selected_both
    assert 'episode_lens' not in selected_both
    
    shutil.rmtree(folder)

if __name__ == "__main__":
    test_get_all_data_subselection()
