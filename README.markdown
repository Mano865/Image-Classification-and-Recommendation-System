# Image Classification and Recommendation System

## Overview
This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch, trained on a dataset from Hugging Face (`blackhole-boys/recomendation-system-dataset-v1`). The model is used to classify images and extract feature embeddings for a recommendation system based on cosine similarity. The code also includes visualization of results, including a confusion matrix and sample predictions.

## Features
- **Data Loading**: Loads a dataset of images and labels from a Parquet file using Dask and converts it to a Pandas DataFrame.
- **Custom Dataset**: Implements a custom `ImageDataset` class for handling image data with transformations.
- **CNN Model**: Defines a `CNN_Classifier` with convolutional layers, batch normalization, and fully connected layers for classification.
- **Training**: Trains the model with early stopping and learning rate scheduling using `ReduceLROnPlateau`.
- **Recommendation System**: Extracts feature embeddings and recommends similar images based on cosine similarity.
- **Evaluation**: Generates a confusion matrix and visualizes sample predictions.
- **Visualization**: Displays the query image and its top-5 similar images, along with a confusion matrix and sample predictions.

## Requirements
To run this code, you need the following Python libraries:
```bash
pip install dask pandas pillow torch torchvision matplotlib tqdm scikit-learn
```

Additionally, you need access to the dataset hosted on Hugging Face (`hf://datasets/blackhole-boys/recomendation-system-dataset-v1`). If the dataset is not publicly accessible, authenticate with Hugging Face or use a local copy.

## Dataset
The dataset is expected to be in Parquet format, with columns:
- `image`: Contains image data as bytes (under `image["bytes"]`).
- `label`: Integer labels for classification.

The code loads the first 40,000 samples using `dd.read_parquet().head(40000)` and converts them to a Pandas DataFrame.

## Project Structure
- **Data Loading**: Loads and preprocesses the dataset.
- **ImageDataset Class**: Converts image bytes to PIL images and applies transformations (resize to 128x128 and convert to tensor).
- **CNN_Classifier Class**: Defines a CNN with 4 convolutional layers, batch normalization, max pooling, and fully connected layers.
- **Training Loop**: Trains the model for up to 30 epochs with early stopping (patience=5) and learning rate scheduling.
- **Recommendation System**: Uses the `forward_features` method to extract embeddings and recommends similar images using cosine similarity.
- **Evaluation and Visualization**:
  - Generates a confusion matrix for the validation set.
  - Visualizes sample predictions with true and predicted labels.
  - Displays a query image and its top-5 similar images.

## Usage
1. **Install Dependencies**:
   Ensure all required libraries are installed (see Requirements).

2. **Run the Script**:
   Execute the script in a Python environment:
   ```bash
   python script.py
   ```
   Replace `script.py` with the name of your Python file.

3. **Expected Output**:
   - Prints the first few rows of the dataset and its size.
   - Displays the device used (CPU or GPU).
   - Shows training and validation loss/accuracy for each epoch.
   - If early stopping is triggered, it will indicate the epoch where training stopped.
   - Displays:
     - A confusion matrix for the validation set.
     - A grid of 10 sample images with true and predicted labels.
     - A query image and its top-5 similar images based on cosine similarity.

## Configuration
- **Hyperparameters**:
  - `IMG_SIZE`: 128 (image resize dimensions).
  - `BATCH_SIZE`: 32 (number of samples per batch).
  - `EPOCHS`: 30 (maximum number of training epochs).
  - `Learning Rate`: 0.001 (initial learning rate for Adam optimizer).
  - `Patience`: 5 (epochs to wait for early stopping).
  - `Scheduler`: Reduces learning rate by a factor of 0.1 if validation loss does not improve for 3 epochs.
- **Model Architecture**:
  - 4 convolutional layers with batch normalization and max pooling.
  - Adaptive average pooling to 4x4 feature maps.
  - Fully connected layers with dropout (0.5) for classification.

## Notes
- **Dataset Size**: The script loads 40,000 samples. For larger datasets, consider loading the full dataset or using Dask’s lazy evaluation, which may require modifying the `ImageDataset` class.
- **Error Handling**: The code does not include explicit error handling for invalid image bytes. You may want to add try-except blocks in the `ImageDataset` class.
- **GPU Support**: The code automatically uses a GPU if available (`torch.cuda.is_available()`).
- **Visualization**: Requires `matplotlib` for plotting. Ensure a display backend is available (e.g., TkAgg for non-interactive environments).

## Future Improvements
- Add data augmentation (e.g., random flips, rotations) to improve model robustness.
- Implement error handling for corrupt image data.
- Save and load the trained model for reuse.
- Add additional evaluation metrics (e.g., precision, recall, F1-score).
- Optimize the recommendation system for larger datasets using approximate nearest neighbors (e.g., FAISS).

## License
This project is for educational purposes and uses a publicly available dataset. Ensure compliance with the dataset’s license and Hugging Face’s terms of use.