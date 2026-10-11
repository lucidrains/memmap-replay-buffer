import urllib.error
import urllib.request

import numpy as np
import pytest

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.frontend import make_server, parse_range, render_page, serve
from memmap_replay_buffer.video import main as video_main, trajectory_to_video

imageio = pytest.importorskip('imageio')
pytest.importorskip('imageio_ffmpeg')

STATS = dict(field = 'image', shape = '16 x 12 x 3', frames = 6, fps = 5)

@pytest.fixture
def video_path(tmp_path):
    frames = np.random.randint(0, 256, (6, 16, 12, 3), dtype = np.uint8)
    return trajectory_to_video(frames, tmp_path / 'video.mp4', fps = 5)

def test_render_page():
    page = render_page(STATS)

    assert '"field": "image"' in page
    assert '__REPLAY_STATS__' not in page
    assert 'window.replay' in page

def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range('bytes=0-9', 100) == (0, 9)
    assert parse_range('bytes=10-', 100) == (10, 99)
    assert parse_range('bytes=-10', 100) == (90, 99)
    assert parse_range('bytes=90-200', 100) == (90, 99)

def test_server(video_path):
    httpd, url = serve(video_path, STATS, port = 0, open_browser = False, block = False)

    try:
        with urllib.request.urlopen(url) as response:
            page = response.read().decode()

            assert response.status == 200
            assert '<video' in page
            assert '"frames": 6' in page

        size = video_path.stat().st_size

        request = urllib.request.Request(url + 'video', headers = dict(Range = 'bytes=0-99'))
        with urllib.request.urlopen(request) as response:
            body = response.read()

            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes 0-99/{size}'
            assert len(body) == 100

        request = urllib.request.Request(url + 'video', headers = dict(Range = 'bytes=-10'))
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert len(response.read()) == 10

        with urllib.request.urlopen(url + 'video') as response:
            assert response.status == 200
            assert len(response.read()) == size

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + 'nope')

        assert error.value.code == 404

    finally:
        httpd.shutdown()
        httpd.server_close()

def test_cli_serve_wiring(monkeypatch, tmp_path, capsys):
    import memmap_replay_buffer.frontend as frontend
    import memmap_replay_buffer.video as video

    calls = dict()

    def fake_serve(video_path, stats, **kwargs):
        calls['video_path'] = video_path
        calls['existed'] = video_path.exists()
        calls['stats'] = stats
        calls['kwargs'] = kwargs

    monkeypatch.setattr(frontend, 'serve', fake_serve)

    folder = tmp_path / 'serve-buffer'

    buffer = ReplayBuffer(
        folder,
        max_episodes = 2,
        max_timesteps = 4,
        fields = dict(image = ('uint8', (8, 8, 3)), action = 'int'),
        circular = True
    )
    buffer.store_episode(image = np.zeros((3, 8, 8, 3), dtype = np.uint8), action = np.zeros(3, dtype = np.int32))

    video.main([str(folder), '--serve', '0', '--no-browser'])

    assert calls['stats']['field'] == 'image'
    assert calls['stats']['shape'] == '8 x 8 x 3'
    assert calls['stats']['dtype'] == 'uint8'
    assert calls['stats']['frames'] == 3
    assert calls['stats']['episode'] == 0
    assert calls['kwargs'] == dict(port = 0, open_browser = False)

    assert calls['existed']
    assert 'wrote' in capsys.readouterr().out

    # the temporary video is cleaned up after serving

    assert not calls['video_path'].exists()

def test_cli_serve_keeps_explicit_output(monkeypatch, tmp_path):
    import memmap_replay_buffer.frontend as frontend
    import memmap_replay_buffer.video as video

    monkeypatch.setattr(frontend, 'serve', lambda *args, **kwargs: None)

    folder = tmp_path / 'serve-output-buffer'

    buffer = ReplayBuffer(
        folder,
        max_episodes = 2,
        max_timesteps = 4,
        fields = dict(image = ('uint8', (8, 8, 3)))
    )
    buffer.store_episode(image = np.zeros((3, 8, 8, 3), dtype = np.uint8))

    output = tmp_path / 'kept.mp4'
    video.main([str(folder), '--serve', '0', '--no-browser', '-o', str(output)])

    assert output.exists()
