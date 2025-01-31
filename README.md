# **Beta Caller**  
**AI-driven climbing guidance using Vision Transformers**  

## 🚀 Overview  
Beta Caller is an AI-driven system that guides rock climber movements (**limb, direction, distance**) using Vision Transformer models. This system is specifically designed to assist climbers with visual impairment by predicting movements and translating them into verbal commands.  

It integrates:  
- **Vision Transformer-based limb location prediction (ClimBEiT)**  
- **YOLOv8 for hold detection**  
- **ViTPose for body pose estimation**  

## 🔧 Features  
✔ **Real-time analysis of climbing movements** using pose estimation and hold detection  
✔ **Using trained models to find the best limb to move and limb location to move to**  
✔ **Translating limb location into a usable command** (direction and distance) for verbal guidance  

## 📂 Repository Structure  

**Main files:**  
- `main.py` – Entry point for running the system  

**Key folders:**  
- `models/` – Contains two models that support pose estimation  
- `utils/` – Core utility functions, organized into submodules:  
  - `data/` – Facilitates data management  
  - `detection/` – Handles hold detection, movement detection, and pose estimation  
  - `prediction/` – Core prediction pipeline:  
    - `limb_prediction/` (predicts which limb will move)  
      - `models.py` – Defines the model class  
      - `preprocessing.py` – Handles image preprocessing  
      - `inference.py` – Performs inference  
      - `limb_prediction.py` – Orchestrates limb prediction  
    - `location_prediction/` (predicts limb movement location, mirrors limb_prediction/)  
    - `call_prediction.py` – Uses text-to-speech to call out movement commands  
    - `command_translation.py` – Maps limb location to a hold and converts to direction/distance  
    - `move_prediction.py` – The main prediction file that orchestrates everything  

## 💻 Installation  
1. Create and activate a virtual environment:
	```bash
	conda create -n beta_caller python=3.9
	conda activate beta_caller
	```
2. Clone the Beta Caller repository:  
   ```bash
   git clone https://github.com/yourusername/beta-caller.git
   cd beta-caller
   ```
3. Install Beta Caller dependencies:
	```bash
	pip install -r requirements.txt
	```
4. Clone the ViTPose repository:  
   ```bash
   git clone https://github.com/JunkyByte/easy_ViTPose.git
   cd easy_ViTPose
   ```
5. Install ViTPose dependencies:
	```bash
	pip install -e .
	pip install -r requirements.txt	
	```

## 🚀 Running Beta Caller
	To run Beta Caller with your camera:
	```bash
	python main.py
	```
	
## 📊 Model Details
	Limb Prediction: Vision Transformer (ClimBEiT) trained on 8,000+ climbing images
	Limb Location Prediction: Vision Transformer (ClimBEiT) trained on 8,000+ climbing images
	Hold Detection: YOLOv8 fine-tuned on a combination of multiple climbing gym holds datasets
	Pose Estimation: ViTPose for body keypoint extraction

## 🤝 Contributing
Contributions are welcome! Please submit an issue or pull request if you’d like to improve Beta Caller.

## 📜 License
MIT License – Feel free to modify and use this project!
