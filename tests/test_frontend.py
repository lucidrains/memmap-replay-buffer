import urllib.error
import urllib.request

import numpy as np
import pytest

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.frontend import make_server, parse_range, render_page, serve
from memmap_replay_buffer.video import main as video_main, trajectory_to_video

imageio = pytest.importorskip('imageio')
pytest.importorskip('imageio_ffmpeg')

STATS = dict(
    folder = '/tmp/buffer',
    episodes = [0],
    frames = 6,
    fps = 5,
    fields = [
        dict(name = 'image', shape = '16 x 12 x 3', dtype = 'uint8'),
        dict(name = 'wrist', shape = '16 x 12 x 3', dtype = 'uint8')
    ]
)

@pytest.fixture
def video_path(tmp_path):
    frames = np.random.randint(0, 256, (6, 16, 12, 3), dtype = np.uint8)
    return trajectory_to_video(frames, tmp_path / 'video.mp4', fps = 5)

def test_render_page():
    page = render_page(STATS)

    assert '"name": "image"' in page
    assert '"name": "wrist"' in page
    assert '__REPLAY_STATS__' not in page
    assert 'window.replay' in page
    assert 'id="timeline"' in page
    assert 'id="step-back"' in page
    assert 'seekToFrame' in page

def test_render_page_legacy_stats():
    page = render_page(dict(field = 'image', shape = '16 x 12 x 3', dtype = 'uint8', frames = 6, fps = 5))

    assert '"fields"' in page
    assert '"name": "image"' in page
    assert '"shape": "16 x 12 x 3"' in page

def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range('bytes=0-9', 100) == (0, 9)
    assert parse_range('bytes=10-', 100) == (10, 99)
    assert parse_range('bytes=-10', 100) == (90, 99)
    assert parse_range('bytes=90-200', 100) == (90, 99)

def test_server(video_path):
    httpd, url = serve(dict(image = video_path, wrist = video_path), STATS, port = 0, open_browser = False, block = False)

    try:
        with urllib.request.urlopen(url) as response:
            page = response.read().decode()

            assert response.status == 200
            assert 'id="replays"' in page
            assert 'video.src' in page
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

        # named videos and a legacy bare path

        with urllib.request.urlopen(url + 'video/wrist') as response:
            assert response.status == 200
            assert response.headers['Content-Type'] == 'video/mp4'

        for missing in ('nope', 'video/nope'):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + missing)

            assert error.value.code == 404

        # a bare path is still accepted and served as a single video

        legacy_httpd, legacy_url = serve(video_path, STATS, port = 0, open_browser = False, block = False)

        try:
            with urllib.request.urlopen(legacy_url + 'video') as response:
                assert response.status == 200
        finally:
            legacy_httpd.shutdown()
            legacy_httpd.server_close()

    finally:
        httpd.shutdown()
        httpd.server_close()

def test_cli_serve_wiring(monkeypatch, tmp_path, capsys):
    import memmap_replay_buffer.frontend as frontend
    import memmap_replay_buffer.video as video

    calls = dict()

    def fake_serve(videos, stats, **kwargs):
        calls['videos'] = videos
        calls['existed'] = videos['image'].exists()
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

    assert set(calls['videos']) == {'image'}
    assert calls['stats']['fields'] == [dict(name = 'image', shape = '8 x 8 x 3', dtype = 'uint8')]
    assert calls['stats']['episodes'] == [0]
    assert calls['stats']['frames'] == 3
    assert calls['kwargs'] == dict(port = 0, open_browser = False)

    assert calls['existed']
    assert 'wrote' in capsys.readouterr().out

    # the temporary video is cleaned up after serving

    assert not calls['videos']['image'].exists()

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
