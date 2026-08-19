<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/jerichd4c/face-recognition-sistem">
    <img src="https://raw.githubusercontent.com/jerichd4c/ReflexJDBC/main/python_logo.svg" alt="Logo" width="80" height="80">
  </a>
</div>


<div align="center">
  <h3 align="center">Real-Time Face Recognition System </h3>

  <p align="center">
    A Python application for registering people, recognizing faces in real-time, and analyzing emotions using OpenCV and DeepFace, with a simple Streamlit web interface and SQLite persistence.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#database-and-persistence">Database and Persistence</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#acknowledgments">Credits</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a comprehensive solution for real-time facial recognition and emotion analysis. It leverages advanced deep learning models to identify registered individuals and determine their emotional state.

Key Features:
- **Facial Recognition** using embeddings (Facenet, ArcFace, or VGG-Face).
- **Emotion Analysis** (7 classes: angry, disgust, fear, happy, sad, surprise, neutral).
- **Fine-tuning** to improve "disgust" detection and temporal smoothing.
- **Interactive Reports** with charts and emotion-based filtering.
- **Full Administration** of people and detection logs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-shield]][Python-url]
* [![OpenCV][OpenCV-shield]][OpenCV-url]
* [![Streamlit][Streamlit-shield]][Streamlit-url]
* [![SQLite][SQLite-shield]][SQLite-url]
* [![DeepFace][DeepFace-shield]][DeepFace-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* **Python 3.10** or higher.
* **Windows** recommended (tested with local camera).
* **Webcam** available.

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/jerichd4c/facial-recognition-cv.git
   cd facial-recognition-cv
   ```
2. Create and activate virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Web Interface (Streamlit)

Run the interactive interface for registration, detection, and reports:
```sh
streamlit run app.py
```

The application features three main sections:
1. **Registration:** Face capture and person enrollment.
2. **Detection:** Native OpenCV window for real-time recognition. (Press `q` to close).
3. **Reports:** Visualization of statistics and recent detections.

### Command Line Interface (CLI)

You can also use the native engine directly:

**Register:**
```powershell
python camera_local.py register --camera 0 --nombre "Firstname" --apellido "Lastname" --email "email@example.com"
```

**Detect:**
```powershell
python camera_local.py detect --camera 0 --threshold 0.6
```

_For more details on CLI parameters, check the script help: `python camera_local.py --help`._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PROJECT STRUCTURE -->
## Project Structure

```text
app.py                # Streamlit interface (Registration, Detection, Reports)
camera_local.py       # Native engine (OpenCV/DeepFace) and CLI
requirements.txt      # Project dependencies
facial_recognition.db # SQLite database (created upon execution)
models/               # Caffe models for opencv-dnn (optional)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATABASE -->
## Database and Persistence

The system uses **SQLite** (`facial_recognition.db`) to store:
- Person information.
- Face embeddings.
- Detection history with detailed emotion data.

When a person is deleted, detection records are preserved (setting the ID to NULL) to maintain statistical integrity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TROUBLESHOOTING -->
## Troubleshooting

- **Camera not detected:** Check the camera index (`--camera`) and system permissions.
- **DeepFace errors:** The first run downloads model weights; this may take several minutes.
- **Low performance:** Try reducing resolution or increasing `infer-interval-ms`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [OpenCV](https://opencv.org/)
* [DeepFace](https://github.com/serengil/deepface)
* [Streamlit](https://streamlit.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[OpenCV-shield]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white&color=5C3EE8
[OpenCV-url]: https://opencv.org/
[Streamlit-shield]: https://img.shields.io/badge/streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[SQLite-shield]: https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white
[SQLite-url]: https://www.sqlite.org/
[DeepFace-shield]: https://img.shields.io/badge/DeepFace-AI-blue?style=for-the-badge
[DeepFace-url]: https://github.com/serengil/deepface
