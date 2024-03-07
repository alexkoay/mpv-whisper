# MPV-Whisper

Adding Whisper audio transcription capabilities to MPV


## Installing

Simply run the following to get started:

```pip install mpv-whisper```

To use the GPU, ensure that the appropriate version of PyTorch is installed.


## Configuration

mpv-whisper searches for configuration in two places, and defaults back to the package-provided default if not found:

- `./mpv-whisper.toml`
- `~/.config/mpv-whisper/config.toml`
- package provided `config.toml`

Refer to the default [`config.toml`](https://github.com/alexkoay/mpv-whisper/blob/master/mpv_whisper/config.toml) for accepted configuration values.

## Roadmap

- Using `dump_cache` or something similar to extract audio instead of running FFmpeg separately

## Credits

- [GhostNaN/whisper-subs](https://github.com/GhostNaN/whisper-subs) for providing pointers on how to add whisper to MPV
