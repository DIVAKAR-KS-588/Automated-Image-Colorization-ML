# Automatic Image Colorization Using Machine Learning

## 1. Project Overview

Automatic Image Colorization is a machine-learning project that converts grayscale images into color images using a Convolutional Neural Network (CNN). The model learns patterns from training images and predicts plausible colors based on image features such as edges, shapes, textures, and lightness.

The project uses Python, PyTorch, the Oxford-IIIT Pet dataset, and the CIE Lab color space. A Google Colab notebook (`.ipynb`) can be used to run the workflow, train the model, and evaluate its output.

**Important:** The generated colors are predictions. They may look realistic, but they are not guaranteed to match the original colors of the photograph.

## 2. Problem Statement

Black-and-white photographs contain brightness and structural information but lack explicit color information. Manually coloring these photographs takes time and requires human effort.

This project develops an automated system that learns from color images and predicts suitable colors for grayscale inputs.

## 3. Objectives

- Convert grayscale photographs into color images automatically.
- Learn image features using a CNN.
- Predict color information from image brightness and structure.
- Evaluate predictions using numerical image-quality metrics.
- Provide a simple interface for uploading and downloading colorized images.

## 4. Applications

- Restoring old black-and-white photographs.
- Creating color previews for historical images.
- Supporting digital media and image-editing workflows.
- Educational demonstrations of machine learning and computer vision.
- Assisting image restoration projects.

Colorization should be reviewed by a person when historical or scientific color accuracy is important.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab / Jupyter Notebook | Running code and experiments |
| PyTorch | Building, training, and running the CNN |
| CNN encoder-decoder | Learning image features and predicting colors |
| Oxford-IIIT Pet dataset | Training and testing with pet images |
| CIE Lab color space | Separating brightness from color |
| NumPy | Numerical operations |
| Pillow | Loading, resizing, and saving images |
| Streamlit | Optional web interface for image upload and download |

## 6. Dataset

The project uses the **Oxford-IIIT Pet dataset**, which contains images of cats and dogs.

The original RGB images are converted into the CIE Lab color space. The model receives the lightness channel as input and learns to predict the two color channels.

The dataset is pet-focused, so the model may perform less effectively on landscapes, vehicles, buildings, and human portraits.

Dataset source: https://www.robots.ox.ac.uk/~vgg/data/pets/

## 7. How the Project Works

1. **Load the dataset:** Read the available color images.
2. **Preprocess images:** Resize images to the model's required dimensions.
3. **Convert color space:** Convert RGB images into CIE Lab.
4. **Prepare input and target:** Use the L channel as input and the a/b channels as the target.
5. **Train the CNN:** Predict the color channels and calculate the difference from the target.
6. **Update model parameters:** Use backpropagation and the Adam optimizer to reduce the loss.
7. **Evaluate the model:** Calculate metrics on test images not used for training.
8. **Generate the output:** Combine the input L channel with predicted a/b channels and convert the result back to RGB.
9. **Display the result:** Show the grayscale input and predicted color image; optionally allow downloading the output through Streamlit.

### Workflow

`RGB Dataset → Lab Conversion → L Channel Input → CNN → Predicted a/b Channels → Lab-to-RGB Conversion → Colorized Image`

## 8. Machine Learning Algorithm

### Convolutional Neural Network (CNN)

CNN stands for **Convolutional Neural Network**. It is a neural-network architecture commonly used for image-processing tasks.

A CNN learns visual patterns from image pixels. Early layers can learn simple patterns such as edges and textures, while deeper layers can learn more complex structures.

### CNN encoder-decoder

The project uses a CNN encoder-decoder design with U-Net-style skip connections, as described in the project implementation.

- **Encoder:** Extracts visual features from the input image.
- **Bottleneck:** Represents higher-level image information.
- **Decoder:** Reconstructs spatial information and predicts the a/b color channels.
- **Skip connections:** Transfer features from encoder layers to corresponding decoder layers, helping preserve image details.

The network predicts two output channels rather than directly predicting three RGB channels.

### Other algorithms and methods involved

- **Supervised learning:** The model learns from input images paired with their known color-channel targets.
- **Backpropagation:** Computes gradients of the loss with respect to model parameters.
- **Adam optimizer:** Updates model parameters using gradient information.
- **L1 loss:** Measures the absolute difference between predicted and target color channels.
- **Train-validation-test split:** Training data updates the model, validation data helps monitor training, and test data measures performance on unseen examples.

The primary learning model is the CNN. Adam, backpropagation, and L1 loss are training methods rather than separate colorization models.

## 9. Color Space: RGB and CIE Lab

### RGB color space

RGB represents an image using three channels:

- R: Red
- G: Green
- B: Blue

### CIE Lab color space

CIE Lab separates lightness from color:

- **L:** Lightness, from dark to bright.
- **a:** Green-to-red color information.
- **b:** Blue-to-yellow color information.

The model receives L and predicts a and b. The predicted channels are then combined with L to reconstruct a color image.

This design lets the network focus on predicting color while preserving the original image's brightness structure.

## 10. Mathematical Formulas

Let:

- \(L\) = input lightness channel.
- \(a,b\) = actual color channels.
- \(\hat{a},\hat{b}\) = predicted color channels.
- \(N\) = number of color-channel values being compared.

### 10.1 L1 Loss (Mean Absolute Error)

L1 loss is the main training loss described in this project.

\[
L_{\text{L1}}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
\]

Here, \(y_i\) is an actual target value and \(\hat y_i\) is the predicted value. In this project, the comparison is between the actual and predicted a/b channels.

**Example calculation:**

Suppose the actual values are \([2,4,6]\) and the predictions are \([3,3,5]\).

\[
L_{\text{L1}}=\frac{|2-3|+|4-3|+|6-5|}{3}
\]

\[
L_{\text{L1}}=\frac{1+1+1}{3}=1
\]

A lower L1 loss means the predictions are closer to the target values on average. It does not by itself guarantee visually pleasing colors.

### 10.2 Mean Squared Error (MSE)

MSE calculates the average squared difference between actual and predicted values.

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

Example, using the same values:

\[
\mathrm{MSE}=\frac{(2-3)^2+(4-3)^2+(6-5)^2}{3}
\]

\[
\mathrm{MSE}=\frac{1+1+1}{3}=1
\]

Squaring gives larger errors more influence on the result. MSE is a general evaluation metric; it is not the main training loss described here.

### 10.3 Peak Signal-to-Noise Ratio (PSNR)

PSNR expresses the relationship between the maximum possible pixel value and the mean squared pixel error.

\[
\mathrm{PSNR}=10\log_{10}\left(\frac{MAX^2}{\mathrm{MSE}}\right)
\]

For RGB images normalized to the range 0–1, \(MAX=1\). If MSE is \(0.01\):

\[
\mathrm{PSNR}=10\log_{10}\left(\frac{1}{0.01}\right)
=20\text{ dB}
\]

A higher PSNR generally indicates a smaller pixel-level error. In the project's evaluation script, PSNR is calculated for one example image, so it should not be reported as an average over the entire test set.

### 10.4 Mean Absolute Error for predicted color channels

The evaluation script calculates the mean normalized absolute error between predicted and target a/b channels:

\[
\mathrm{MAE}_{ab}=
\frac{1}{N}\sum_{i=1}^{N}|c_i-\hat c_i|
\]

Here, \(c_i\) represents an actual normalized a/b value and \(\hat c_i\) represents its prediction. The implementation averages this error over the test images.

A lower value means the predicted color channels are numerically closer to their targets.

**Note:** The numerical values of L1 loss and MAE can be identical when calculated over the same values using the same normalization. In this project, training loss and test MAE may differ because they are calculated on different data and at different stages.

## 11. Model Training and Calculations

During training, the following operations occur:

1. The model receives a batch of L-channel images.
2. The CNN produces predicted a/b channels.
3. The loss function compares predictions with the known target channels.
4. Backpropagation calculates gradients of the loss.
5. Adam updates the model parameters.
6. The process repeats over batches and epochs.

An **epoch** is one complete pass through the training dataset. A **batch** is a group of images processed together before an optimizer update.

For example, if a training set contains 800 images and the batch size is 8, then one epoch contains approximately:

\[
\text{Batches per epoch}=\frac{800}{8}=100
\]

This example illustrates the calculation only; it is not a claim about the actual dataset size used in your run.

The optimizer changes model weights to reduce the training loss. The trained weights are saved in `outputs/colorizer.pth` in the project implementation.

## 12. Evaluation

The project's evaluation script uses a separate Oxford-IIIT Pet test split. It evaluates up to 100 images and reports:

- Number of test images evaluated.
- Mean normalized a/b MAE.
- Example-image RGB PSNR.
- Sample original, grayscale, and colorized images.

Run the evaluation and record the actual output from your experiment. Do not use a quick training run's metrics as final results.

| Metric | What it measures | Preferred direction |
|---|---|---|
| L1 loss | Absolute color-channel prediction error | Lower |
| MAE | Average absolute difference | Lower |
| MSE | Average squared difference | Lower |
| PSNR | Pixel-level reconstruction quality | Higher |

No single metric fully captures whether the predicted colors look realistic. Visual inspection is also important.

=====================================================================================================================================================================

## 14. Running the Google Colab Notebook

1. Open the notebook in Google Colab.
2. Select a suitable runtime; a GPU may accelerate training if available.
3. Run the installation and import cells.
4. Load or download the dataset.
5. Run the preprocessing cells.
6. Execute the model definition and training cells.
7. Run the evaluation cells.
8. Display the original grayscale and colorized outputs.
9. Save your notebook with outputs if you want the results visible on GitHub.

Cell names and order depend on the actual notebook. Follow the cells provided in your notebook.

## 15. Limitations

- A grayscale image does not uniquely determine the original colors.
- The Oxford-IIIT Pet dataset focuses on cats and dogs.
- Predictions can be pale, blurry, or incorrect.
- A short training run may not learn enough visual patterns.
- Pixel-level metrics do not fully measure perceptual quality.
- GPU availability, image resolution, dataset size, and training duration affect experiment speed and results.

## 16. Future Enhancements

- Train with a broader dataset containing landscapes, buildings, people, and everyday objects.
- Experiment with longer training and suitable hyperparameters.
- Compare the current CNN with alternative architectures.
- Explore perceptual losses or adversarial methods.
- Evaluate with additional metrics such as SSIM and LPIPS when supported by the implementation.
- Improve the interface with before-and-after comparison and downloadable results.

## 17. Conclusion

This project demonstrates how supervised machine learning can be used to colorize grayscale photographs. A CNN learns a mapping from image lightness to color channels in CIE Lab space. L1 loss and the Adam optimizer support training, while MAE and PSNR help evaluate numerical output quality.

The project provides a practical introduction to CNNs, image preprocessing, color spaces, model training, and computer-vision evaluation using Python and PyTorch.

## 18. Author and Project Results

**Project:** Automatic Image Colorization Using Machine Learning  
**Platform:** Google Colab / Jupyter Notebook  
**Primary model:** CNN encoder-decoder  
**Dataset:** Oxford-IIIT Pet
