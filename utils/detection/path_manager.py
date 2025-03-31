from pathlib import Path

# get the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# model paths
MODELS_DIR = PROJECT_ROOT / 'models'
VITPOSE_MODEL_PATH = str(MODELS_DIR / 'vitpose-s-coco.pth')
YOLO_MODEL_PATH = str(MODELS_DIR / 'yolov8s.pt')
LIMB_LOC_MODEL_PATH = str(MODELS_DIR / 'limb_xy.keras')

# print(PROJECT_ROOT)
# print(MODELS_DIR)
# print(VITPOSE_MODEL_PATH)
# print(YOLO_MODEL_PATH)
