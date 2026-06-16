# TBtector

TBtector is a deep learning project aimed at diagnosing and classifying lung conditions from medical images. This repository contains the code for training, evaluating, and running inference with our machine learning models.

## Project Structure

* `data/`: Directory for storing the datasets (images and labels). *Note: Datasets are excluded from version control.*
* `venv/`: Python virtual environment.
* `models/`: Directory where trained model weights (`.pth` / `.h5`) are saved.

## Requirements

To run this project, you need Python installed on your machine. The project uses PyTorch / TensorFlow for model training and inference.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd TBtector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   * On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   * On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

*(Add instructions here on how to train the model, evaluate it, or run inference on new images.)*

## License

MIT License