from pathlib import Path

# --- Video ---
MAX_HEIGHT = 480                 
OUTPUT_DIR = Path("output")
VIDEO_PATH = OUTPUT_DIR / "clip.mp4"
EVIDENCE_PATH = OUTPUT_DIR / "max_aforo.jpg"     
ANNOTATED_PATH = OUTPUT_DIR / "annotated.mp4"    

# --- Visualización en tiempo real ---
SHOW_LIVE = True          
SAVE_ANNOTATED = True     

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_NAME = str(MODELS_DIR / "yolo11x.pt")
     
CONF_THRESHOLD = 0.42
IMG_SIZE = 480                 
PERSON_CLASS_ID = 0              

# --- Velocidad ---
FRAME_SKIP = 1