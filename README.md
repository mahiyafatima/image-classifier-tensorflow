# Image Classifier with TensorFlow (CIFAR-10)

A simple deep learning project using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. It classifies custom images (like cars, trucks, cats, etc.) using a pre-trained TensorFlow model.

## Dataset

- CIFAR-10: 60,000 images across 10 categories  
- Classes: `Plane`, `Car`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`

##  Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Project Structure

image-classifier-tensorflow/
├── classifier.py # Main prediction script
├── Image Classification/
│ ├── image_classifier.keras # Pretrained model
│ ├── car.jpg # Sample test image
├── README.md


##  How It Works

- Loads CIFAR-10 dataset (optional for reference)
- Loads pre-trained `.keras` model
- Reads and resizes a custom image (e.g., car.jpg)
- Normalizes the image
- Runs prediction and displays the predicted class

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mahiyafatema/image-classifier-tensorflow.git
   cd image-classifier-tensorflow
### Install required packages:
pip install tensorflow opencv-python matplotlib numpy
Run the script:

### python classifier.py
 Make sure your .keras model file and image (car.jpg) are in the correct folder as specified in the script.

### Example Output
Prediction is Car
