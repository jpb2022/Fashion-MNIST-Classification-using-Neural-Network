# Fashion-MNIST-Classification-using-Neural-Network

## Project Overview

This project implements a neural network to classify images from the Fashion-MNIST dataset, which contains grayscale images of 10 fashion categories. The solution demonstrates a complete machine learning workflow including data preparation, model development, training, evaluation, and optimization.

## Dataset

The Fashion-MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- 10 classes of fashion items:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## Model Architecture

The neural network features:
- Input layer (784 neurons)
- Multiple hidden layers (512, 256, 128 neurons)
- Batch normalization after each hidden layer
- ReLU activation functions
- Dropout layers for regularization
- Output layer (10 neurons) with LogSoftmax

## Key Features

1. **Data Preparation**
   - Normalization of pixel values
   - Train/validation/test splits
   - Data loaders with batching

2. **Model Training**
   - Custom training loop with validation
   - Early stopping
   - Learning rate scheduling
   - Loss and accuracy tracking

3. **Evaluation**
   - Test set evaluation
   - Confusion matrix visualization
   - Prediction examples

4. **Optimization**
   - Hyperparameter tuning
   - Regularization techniques
   - Performance monitoring

## Results

The model achieves:
- Training accuracy: 93.69%
- Validation accuracy: 91.05%
- Test accuracy: 90.30%

## Requirements

To run this notebook, you'll need:
- Python 3.6+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- seaborn

## How to Use

1. Clone the repository
2. Install required packages
3. Run the Jupyter notebook sequentially
4. The notebook will:
   - Download and prepare the dataset
   - Define and train the model
   - Evaluate performance
   - Visualize results

## Key Learnings

Through this project, I gained experience with:
- Neural network architecture design
- Training optimization techniques
- Overfitting prevention
- Model evaluation best practices
- Hyperparameter tuning

## Files

- `FashionMNIST_Classification.ipynb`: Main notebook with complete solution
- `README.md`: This documentation file

## Future Improvements

Potential enhancements include:
- Experimenting with convolutional layers
- Implementing data augmentation
- Trying different optimizer configurations
- Exploring model pruning techniques

## License

This project is open source and available under the MIT License.
