# Real-time Action Recognition using Pose Estimation

This project builds a web application capable of recognizing user actions in real-time via a computer's webcam, utilizing extracted skeleton data and an ST-GCN deep learning model.

## 🌟 Overview

The application uses the camera for video input, extracts the human skeleton using MediaPipe, and then feeds the sequence of skeletons into a fine-tuned ST-GCN model to classify the action into one of the 7 target classes:

1. Walking  
2. Running  
3. Jumping  
4. Standing Up  
5. Carrying  
6. Lying Down  
7. Unknown (if confidence is low)

The prediction result is displayed directly on the web interface.

## ✨ Key Features

* **Real-time** action recognition from webcam.  
* Utilizes the powerful **ST-GCN** (Spatial Temporal Graph Convolutional Network) model for skeleton data.  
* Fine-tuned on the **NTU RGB+D 60** dataset combined with **custom user-collected data**.  
* Simple **web interface** displaying the video feed and prediction results.  
* Uses **MediaPipe** for fast and lightweight skeleton extraction.  
* Backend built with **FastAPI** and **Socket.IO** for efficient real-time communication.

## 📄 Project Report & Slides

* **Project Report:** [View Report on Google Drive](https://drive.google.com/file/d/1V8JNsceJRXDEHlPry8_nMztem8zQKGW9/view?usp=sharing)  
* **Presentation Slides:** [View Slides on Google Docs](https://docs.google.com/presentation/d/11nguYARliARM0Ey1O6FNSeNUVGVmT7mG/edit?usp=sharing&ouid=103546190446029251353&rtpof=true&sd=true)  
* **Presentation Video (English):** [Watch on YouTube](https://youtu.be/gI9Wqh7hmak)

## 🛠️ Tech Stack

* **Language:** Python 3.9+  
* **Deep Learning Framework:** PyTorch  
* **Image/Video Processing:** OpenCV, MediaPipe  
* **Web Backend:** FastAPI, Uvicorn, Python-SocketIO  
* **Web Frontend:** HTML, CSS, JavaScript, Socket.IO Client  
* **Other Libraries:** NumPy, Scikit-learn, Matplotlib, Seaborn, tqdm  
* **Data:** NTU RGB+D 60 Skeletons, Custom collected video data.

## 📂 Directory Structure

```plaintext
pose_estimation_project/
│
├── 📂 app/              # Web application code (FastAPI, HTML, CSS, JS)
├── 📂 configs/          # Configuration files (config.yaml - optional)
├── 📂 data/             # Project data
│   ├── 📂 raw/          # Raw data (custom videos, original NTU skeletons)
│   └── 📂 processed/    # Processed data (filtered NTU skeletons, custom skeletons)
├── 📂 notebooks/        # Jupyter Notebooks (data exploration, model evaluation)
├── 📂 scripts/          # Utility scripts (filter NTU data, process custom videos)
├── 📂 src/              # Main source code (model, training, inference, pose extractor)
│   ├── 📂 data_processing/ # (Currently unused, kept for structure)
│   ├── 📂 models/
│   │   └── 📂 utils/
│   ├── __init__.py
│   ├── evaluate_model.py # Evaluation script (replaced by evaluate.ipynb)
│   ├── inference.py
│   ├── pose_extractor.py
│   └── train.py
├── 📂 weights/          # Stores model weights (.pt) and training history (.json)
├── 📂 evaluation_results/ # Stores evaluation results (plots, report)
│
├── .gitignore
├── README.md            # This file
└── requirements.txt     # Python library dependencies
