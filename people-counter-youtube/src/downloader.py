from pathlib import Path

import yt_dlp

from .utils import get_logger

log = get_logger()


def download_video(url: str, output_path: Path, max_height: int = 480) -> Path:
    """Descarga sólo el stream de video en baja resolución."""
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]/"
            f"bestvideo[height<={max_height}]/bestvideo[ext=mp4]/bestvideo"
        ),
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
    }
    log.info(f"Descargando video: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    log.info(f"Video guardado en {output_path}")
    return output_path
