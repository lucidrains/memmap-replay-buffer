from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

# helpers

def to_uint8(frames: np.ndarray):
    frames = np.asarray(frames)

    if frames.dtype == np.uint8:
        return frames

    if np.issubdtype(frames.dtype, np.floating):
        return (frames.clip(0., 1.) * 255.).round().astype(np.uint8)

    raise TypeError(f'cannot convert frames of dtype {frames.dtype} to uint8')

def frames_to_sheet(frames: np.ndarray, rows = 2, columns = 5):
    """ evenly sample frames into a rows x columns grid """
    frames = np.asarray(frames)

    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis = -1)

    indices = np.linspace(0, len(frames) - 1, rows * columns).astype(int)

    return np.concatenate([
        np.concatenate([frames[index] for index in indices[row * columns:(row + 1) * columns]], axis = 1)
        for row in range(rows)
    ], axis = 0)

# main

def trajectory_to_video(
    frames: np.ndarray,
    path: str | Path,
    fps: int = 30,
    **kwargs
):
    """ write a trajectory of frames to a video or animated image

    frames is (time, height, width, channels) or (time, height, width), uint8 or float in [0., 1.]
    the format is inferred from the suffix, e.g. `.mp4` (needs imageio-ffmpeg) or `.gif`
    """
    import imageio

    frames = to_uint8(frames)

    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis = -1)

    # channel-first (time, 3, height, width) -> (time, height, width, 3)

    if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
        frames = np.moveaxis(frames, 1, -1)

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.gif':
        kwargs.setdefault('duration', 1000. / fps)
    else:
        kwargs.setdefault('fps', fps)

        if suffix in ('.mp4', '.mkv', '.mov'):
            # do not auto-resize frames to macroblock multiples

            kwargs.setdefault('macro_block_size', None)

    imageio.mimwrite(str(path), frames, **kwargs)

    return path

def open_buffer(folder: str | Path):
    """ open a replay buffer folder read-only, detecting the h5py backend """
    from memmap_replay_buffer import ReplayBuffer

    folder = Path(folder)

    if (folder / 'data.h5').exists():
        from memmap_replay_buffer import ReplayBufferH5PY
        return ReplayBufferH5PY.from_folder(folder, read_only = True)

    return ReplayBuffer.from_folder(folder, read_only = True)

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog = 'inspect-replay-buffer',
        description = 'render one episode of image frames stored in a replay buffer to a video (and optionally a contact sheet)'
    )
    parser.add_argument('folder', help = 'replay buffer folder')
    parser.add_argument('-e', '--episode', type = int, default = 0, help = 'episode (buffer row) to render')
    parser.add_argument('-f', '--field', default = None, help = 'image field name - auto-detected if there is exactly one image-like field')
    parser.add_argument('--fps', type = int, default = 30, help = 'frames per second')
    parser.add_argument('-o', '--output', default = None, help = 'output video path - defaults to `<folder>-episode-<n>.mp4`')
    parser.add_argument('--scale', type = int, default = 1, help = 'downsample frames by an integer factor')
    parser.add_argument('--sheet', default = None, help = 'also write a contact sheet of frames to this png path')
    parser.add_argument('--rows', type = int, default = 2, help = 'contact sheet rows')
    parser.add_argument('--columns', type = int, default = 5, help = 'contact sheet columns')
    parser.add_argument('--serve', type = int, nargs = '?', const = 8000, default = None, help = 'serve a local frontend on this port (default 8000)')
    parser.add_argument('--no-browser', action = 'store_true', help = 'do not open the frontend in a browser')

    args = parser.parse_args(argv)

    try:
        buffer = open_buffer(args.folder)
        field = args.field or buffer.default_image_field()
        frames = buffer.episode_frames(args.episode, field = field)
    except ValueError as e:
        parser.error(str(e))

    if args.scale > 1:
        frames = frames[:, ::args.scale, ::args.scale]

    temp_dir = None

    if args.output:
        output = Path(args.output)
    elif args.serve is not None:
        temp_dir = Path(tempfile.mkdtemp(prefix = 'inspect-replay-buffer-'))
        output = temp_dir / 'video.mp4'
    else:
        output = Path(f'{Path(args.folder).resolve().name}-episode-{args.episode}.mp4')

    path = trajectory_to_video(frames, output, fps = args.fps)

    print(f'wrote {path} ({len(frames)} frames of {frames.shape[1:]} at {args.fps} fps)')

    if args.sheet:
        import imageio

        sheet_path = Path(args.sheet)
        imageio.imwrite(str(sheet_path), frames_to_sheet(frames, args.rows, args.columns))

        print(f'wrote {sheet_path} ({args.rows}x{args.columns} contact sheet)')

    if args.serve is not None:
        from memmap_replay_buffer.frontend import serve

        stats = dict(
            folder = str(Path(args.folder).resolve()),
            field = field,
            shape = ' x '.join(str(dim) for dim in frames.shape[1:]),
            dtype = str(frames.dtype),
            episode = args.episode,
            episodes = buffer.num_episodes,
            frames = len(frames),
            fps = args.fps,
            scale = args.scale
        )

        serve(path, stats, port = args.serve, open_browser = not args.no_browser)

        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors = True)

if __name__ == '__main__':
    main()
