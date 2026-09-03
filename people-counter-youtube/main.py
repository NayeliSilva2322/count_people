import sys
from pathlib import Path  # <- estaba "from zipfile import Path", mal

import config
from src.counter import count_people
from src.downloader import download_video
from src.utils import ensure_dir, get_logger

log = get_logger()


def main(url: str):
    ensure_dir(config.OUTPUT_DIR)

    # --- Medida de emergencia: pedir el archivo local en vez de bajar de YouTube ---
    local_path = input("Ruta del video local (Enter para descargar de YouTube): ").strip()
    if local_path:
        video_path = Path(local_path)
    else:
        video_path = download_video(url, config.VIDEO_PATH, config.MAX_HEIGHT)
    # --------------------------------------------------------------------------------

    result = count_people(
        video_path=video_path,
        model_name=config.MODEL_NAME,
        conf=config.CONF_THRESHOLD,
        img_size=config.IMG_SIZE,
        frame_skip=config.FRAME_SKIP,
        person_class_id=config.PERSON_CLASS_ID,
        save_evidence_path=config.EVIDENCE_PATH,
        save_video_path=config.ANNOTATED_PATH if config.SAVE_ANNOTATED else None,
        show_live=config.SHOW_LIVE,
    )

    log.info(
        f"Aforo máximo: {result['max_count']} personas "
        f"({result['frames_processed']} frames analizados de {result['total_frames']})"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <url_de_youtube>")
        sys.exit(1)
    main(sys.argv[1])