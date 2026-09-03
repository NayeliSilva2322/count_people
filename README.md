# People Counter — Conteo de Aforo con YOLO

Pipeline que analiza un video (de YouTube o local) y cuenta el aforo máximo simultáneo de personas usando un modelo YOLO de detección de cabezas, con tracking para mantener el conteo estable cuando la gente se mueve u ocluye entre sí.

## Requisitos

- Python 3.10+
- Dependencias:
  ```bash
  pip install ultralytics opencv-python torch yt-dlp
  ```

## Configuración

Todo se ajusta en `config.py`:

| Variable | Qué hace |
|---|---|
| `MODEL_NAME` | Ruta al checkpoint YOLO (`.pt`) |
| `CONF_THRESHOLD` | Confianza mínima para aceptar una detección |
| `IMG_SIZE` | Resolución de inferencia |
| `FRAME_SKIP` | Cada cuántos frames se corre la detección (1 = todos) |
| `PERSON_CLASS_ID` | ID de la clase a contar |
| `VIDEO_PATH` / `EVIDENCE_PATH` / `ANNOTATED_PATH` | Rutas de salida |
| `SHOW_LIVE` | Mostrar ventana en vivo mientras procesa |

## Uso

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXX"
```

Al correrlo te va a preguntar:

```
Ruta del video local (Enter para descargar de YouTube):
```

- Si ya tenés el video descargado, escribí la ruta al archivo y Enter.
- Si dejás vacío y apretás Enter, intenta descargarlo de YouTube con la URL pasada por argumento.

Mientras procesa, si `SHOW_LIVE=True`, se abre una ventana con el conteo en vivo — presioná **`q`** (con la ventana en foco) para cortar antes de que termine el video.

## Salidas

- **`output/max_aforo.jpg`** — foto del frame con el aforo máximo detectado.
- **`output/annotated.mp4`** — video completo con las cajas dibujadas (si `SAVE_ANNOTATED=True`).
- En consola, al final: aforo máximo y cuántos frames se analizaron.

## Problemas comunes

- **Error de yt-dlp ("Sign in to confirm you're not a bot")**: usá un video local en vez de descargar, o pasale cookies del navegador a yt-dlp (`--cookies-from-browser firefox`).
- **La tecla `q` no cierra la ventana**: hacé clic sobre la ventana de video para darle foco antes de apretarla.
