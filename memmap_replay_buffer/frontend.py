from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# helpers

def normalize_stats(stats: dict):
    """ fold the legacy single-field stats keys into a `fields` list """
    stats = dict(stats)

    if 'fields' not in stats and 'field' in stats:
        stats['fields'] = [dict(
            name = stats.pop('field'),
            shape = stats.pop('shape', ''),
            dtype = stats.pop('dtype', '')
        )]

    return stats

def normalize_videos(videos):
    if isinstance(videos, dict):
        return {str(name): Path(path) for name, path in videos.items()}

    return dict(video = Path(videos))

def content_type(path: Path):
    return mimetypes.guess_type(path.name)[0] or 'video/mp4'

def render_page(stats: dict):
    template_path = Path(__file__).parent / 'frontend' / 'index.html'
    return template_path.read_text().replace('__REPLAY_STATS__', json.dumps(normalize_stats(stats)))

def parse_range(range_header: str | None, size: int):
    if not range_header or not range_header.startswith('bytes='):
        return None

    start_str, _, end_str = range_header[len('bytes='):].split(',')[0].partition('-')

    if not start_str:
        # suffix range, e.g. bytes=-500
        return max(size - int(end_str), 0), size - 1

    return int(start_str), min(int(end_str), size - 1) if end_str else size - 1

# server

class ReplayBufferHandler(BaseHTTPRequestHandler):
    video_paths: dict[str, Path]
    page: str

    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/':
            return self.send_page()

        video_path = self.video_for_path(path)

        if video_path is not None:
            return self.send_video(video_path)

        self.send_error(404)

    def video_for_path(self, path):
        # `/video` serves the first video for backward compatibility

        if path == '/video':
            return next(iter(self.video_paths.values()), None)

        if path.startswith('/video/'):
            return self.video_paths.get(unquote(path[len('/video/'):]))

        return None

    def send_page(self):
        body = self.page.encode()

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_video(self, video_path):
        size = video_path.stat().st_size
        byte_range = parse_range(self.headers.get('Range'), size)

        start, end = byte_range if byte_range else (0, size - 1)
        is_partial = byte_range is not None

        self.send_response(206 if is_partial else 200)
        self.send_header('Content-Type', content_type(video_path))
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')

        if is_partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')

        self.end_headers()

        # stream the selected slice, seeking works via repeated range requests

        with open(video_path, 'rb') as f:
            f.seek(start)
            remaining = end - start + 1

            while remaining > 0:
                chunk = f.read(min(remaining, 1 << 20))

                if not chunk:
                    break

                self.wfile.write(chunk)
                remaining -= len(chunk)

def make_server(videos: str | Path | dict[str, Path], stats: dict, host = '127.0.0.1', port = 8000):
    handler = type('Handler', (ReplayBufferHandler,), dict(
        video_paths = normalize_videos(videos),
        page = render_page(stats)
    ))

    return ThreadingHTTPServer((host, port), handler)

def serve(videos: str | Path | dict[str, Path], stats: dict, host = '127.0.0.1', port = 8000, open_browser = True, block = True):
    httpd = make_server(videos, stats, host, port)
    url = f'http://{host}:{httpd.server_address[1]}/'

    print(f'serving replay at {url} (ctrl-c to stop)')

    if open_browser:
        threading.Thread(target = webbrowser.open, args = (url,), daemon = True).start()

    if not block:
        threading.Thread(target = httpd.serve_forever, daemon = True).start()
        return httpd, url

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

    return httpd, url
