import numpy as np
import pytest
import torch

from memmap_replay_buffer import ReplayBuffer, trajectory_to_video
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY
from memmap_replay_buffer.video import frames_to_sheet, main as video_main, to_uint8

imageio = pytest.importorskip('imageio')
pytest.importorskip('imageio_ffmpeg')

def read_video(path):
    return np.asarray(imageio.mimread(str(path)))

def test_uint8_conversion():
    assert to_uint8(np.zeros((2, 4, 4, 3), dtype = np.uint8)).dtype == np.uint8

    floats = to_uint8(np.array([0., 0.5, 1., 2., -1.], dtype = np.float32))
    assert floats.tolist() == [0, 128, 255, 255, 0]

    with pytest.raises(TypeError, match = 'dtype'):
        to_uint8(np.zeros(3, dtype = np.int32))

def test_trajectory_to_video_mp4(tmp_path):
    frames = np.random.randint(0, 256, (5, 16, 12, 3), dtype = np.uint8)

    path = trajectory_to_video(frames, tmp_path / 'trajectory.mp4', fps = 10)

    assert path.exists() and path.stat().st_size > 0
    assert read_video(path).shape == frames.shape

def test_trajectory_to_video_grayscale_float(tmp_path):
    frames = np.zeros((4, 16, 12), dtype = np.float32)
    frames[::2] = 1.

    path = trajectory_to_video(frames, tmp_path / 'grayscale.gif', fps = 5)

    read = read_video(path)

    assert read.shape == (4, 16, 12, 3)
    assert read[0].mean() > read[1].mean()

def test_trajectory_to_video_channel_first(tmp_path):
    frames = np.random.randint(0, 256, (4, 3, 16, 12), dtype = np.uint8)

    path = trajectory_to_video(frames, tmp_path / 'channel_first.mp4', fps = 5)

    assert read_video(path).shape == (4, 16, 12, 3)

@pytest.fixture(params = [ReplayBuffer, ReplayBufferH5PY], ids = ['memmap', 'h5py'])
def buffer(request, tmp_path):
    buffer = request.param(
        tmp_path,
        max_episodes = 4,
        max_timesteps = 10,
        fields = dict(image = ('uint8', (8, 8, 3)), action = 'int'),
        circular = True
    )

    yield buffer

    if hasattr(buffer, 'file'):
        buffer.file.close()

def test_save_video_from_buffer(buffer, tmp_path):
    frames = (np.arange(4 * 8 * 8 * 3).reshape(4, 8, 8, 3) % 255).astype(np.uint8)

    buffer.store_episode(image = frames, action = np.zeros(4, dtype = np.int32))

    path = buffer.save_video(tmp_path / 'episode.mp4', fps = 5)

    assert path.exists()
    assert read_video(path).shape == (4, 8, 8, 3)

    explicit = buffer.save_video(tmp_path / 'explicit.mp4', episode_index = 0, field = 'image', fps = 5)
    assert explicit.exists()

def test_frames_to_sheet():
    frames = np.zeros((7, 4, 6, 3), dtype = np.uint8)

    assert frames_to_sheet(frames, rows = 2, columns = 3).shape == (8, 18, 3)
    assert frames_to_sheet(frames, rows = 1, columns = 8).shape == (4, 48, 3)

    grayscale = np.zeros((5, 4, 6), dtype = np.uint8)
    assert frames_to_sheet(grayscale, rows = 1, columns = 5).shape == (4, 30, 3)

def test_cli(tmp_path, capsys):
    folder = tmp_path / 'cli-buffer'

    buffer = ReplayBuffer(
        folder,
        max_episodes = 4,
        max_timesteps = 6,
        fields = dict(image = ('uint8', (8, 8, 3)), action = 'int'),
        circular = True
    )

    for length in (5, 3):
        buffer.store_episode(
            image = np.random.randint(0, 256, (length, 8, 8, 3), dtype = np.uint8),
            action = np.zeros(length, dtype = np.int32)
        )

    output = tmp_path / 'cli.mp4'
    sheet = tmp_path / 'cli.png'

    video_main([
        str(folder),
        '--episode', '1', '--field', 'image', '--fps', '5',
        '--output', str(output),
        '--sheet', str(sheet), '--rows', '2', '--columns', '3'
    ])

    assert output.exists()
    assert read_video(output).shape == (3, 8, 8, 3)
    assert imageio.v2.imread(str(sheet)).shape == (16, 24, 3)

    assert 'wrote' in capsys.readouterr().out

def test_cli_h5py(tmp_path):
    folder = tmp_path / 'cli-h5py'

    buffer = ReplayBufferH5PY(
        folder,
        max_episodes = 2,
        max_timesteps = 4,
        fields = dict(image = ('uint8', (8, 8, 3)))
    )
    buffer.store_episode(image = np.zeros((3, 8, 8, 3), dtype = np.uint8))
    buffer.file.close()

    output = tmp_path / 'h5py.mp4'
    video_main([str(folder), '-o', str(output), '--fps', '5'])

    assert read_video(output).shape == (3, 8, 8, 3)

def test_cli_scale_and_errors(tmp_path):
    folder = tmp_path / 'cli-scale'

    buffer = ReplayBuffer(
        folder,
        max_episodes = 2,
        max_timesteps = 4,
        fields = dict(image = ('uint8', (8, 8, 3)))
    )
    buffer.store_episode(image = np.zeros((3, 8, 8, 3), dtype = np.uint8))

    output = tmp_path / 'scaled.mp4'
    video_main([str(folder), '-o', str(output), '--scale', '2', '--fps', '5'])

    assert read_video(output).shape == (3, 4, 4, 3)

    with pytest.raises(SystemExit):
        video_main([str(tmp_path / 'missing')])

    with pytest.raises(SystemExit):
        video_main([str(folder), '--episode', '9'])

    with pytest.raises(SystemExit):
        video_main([str(folder), '--field', 'nonexistent'])

def test_save_video_errors(buffer, tmp_path):
    with pytest.raises(ValueError, match = 'out of range'):
        buffer.save_video(tmp_path / 'missing.mp4')

    buffer.store_episode(image = np.zeros((2, 8, 8, 3), dtype = np.uint8), action = np.zeros(2, dtype = np.int32))

    with pytest.raises(ValueError, match = 'invalid field'):
        buffer.save_video(tmp_path / 'bad_field.mp4', field = 'nonexistent')

    ambiguous = ReplayBuffer(
        tmp_path / 'ambiguous',
        max_episodes = 2,
        max_timesteps = 4,
        fields = dict(image = ('uint8', (8, 8, 3)), latent = ('float', (4, 4)))
    )

    with pytest.raises(ValueError, match = 'multiple'):
        ambiguous.save_video(tmp_path / 'ambiguous.mp4')
