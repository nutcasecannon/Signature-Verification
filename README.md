# Offline Signature Verification Using:
(1) CNN inspired by InceptionV1 
(2) Using SURF and SVM

Reference Research Paper : DOI:10.1109/ICIIP47207.2019.8985925

The code is run on Kaggle using TPU for faster computation.

Datasets Used : CEDAR, BHSig-260 (Hindi and Bengali)
https://www.kaggle.com/datasets/ankita22053139/cedarbhsig-260

ICDAR2009
https://www.kaggle.com/datasets/ishaniroy03/icdar-2009

-------------------------------------------------------------------------------------------------------
# Dataset Description

CEDAR
The dataset utilized for this offline signature verification study is the CEDAR signature dataset, a widely recognized benchmark in signature verification research. This dataset comprises a total of 2,640 images, collected from 55 persons. Each individual contributed 24 genuine signatures, amounting to 1,320 authentic samples, along with 24 forged signatures, leading to an equal number of 1,320 forged samples. All images are in .png format. The forged signatures were carefully created to mimic the genuine ones, making the dataset highly suitable for evaluating the robustness of signature verification systems.

BHSig260 signature dataset
The BHSig260 dataset is composed of two subsets: BHsig-B (Bengali) and BHsig-H (Hindi). The BHSig260 dataset is used for research in offline signature verification, specifically for multilingual scenarios.
§BHSig260 (Bengali) signature dataset contains signatures from 100 persons each having 24 authentic and 30 forged signatures, totaling 2,400 authentic and 3,000 forged signatures.
§BHSig260 (Hindi) signature dataset contains signatures from 160 persons each having 24 authentic and 30 forged signatures, totaling 3,840 authentic and 4,800 forged signatures.

ICDAR2009

The ICDAR 2009 Signature Verification Competition (SigComp2009) dataset, also known as "NFI-online" and "NFI-offline," contains both online and offline signature samples, including genuine signatures from 100 individuals and forgeries from 33 individuals. The online dataset captures dynamic handwriting movement, while the offline dataset contains static images. This dataset is recognized for its complexity due to the presence of skilled forgeries, and it includes variability in writing styles, pressure, and orientation, making it ideal for testing the robustness of signature verification systems.

-------------------------------------------------------------------------------------------------------
# Methodology

# Method 1:

This methodology for offline handwritten signature verification utilizes a Convolutional Neural Network (CNN) inspired by the Inception V1 (GoogleNet) architecture. The model adopts a writer-independent approach, treating genuine and forged signatures as distinct classes.

InceptionSVGNet Architecture
The architecture of InceptionSVGNet builds upon the principles of Inception V1, emphasizing parallel convolutional layers with varying kernel sizes to extract multi-scale features effectively. The model incorporates multiple filters at each layer level, enhancing feature diversity without significantly increasing depth.

1.Inception Modules: Each Inception module consists of four parallel convolutional layers with kernel sizes of 16×16, 8×8, 4×4, and 2×2. These layers operate simultaneously on the input data, extracting features at different scales. To create a single feature map, the outputs from these layers are concatenated along the channel dimension.
2.Layer Configuration: The network begins with an initial convolutional block that generates 64 feature maps from the input tensor. Three successive Inception modules follow, with increasing filter counts (16, 24, and 32 filters in each module) and decreasing spatial dimensions through average pooling layers. A final convolutional layer with 16 filters further refines the feature representation before flattening.
3.Classification : After flattening the feature maps, two connected dense layers are used for high-level feature learning. Dropout regularization (rate = 0.5) is applied between these layers to prevent overfitting. Finally, a softmax layer outputs probabilities for the two classes: genuine and forged signatures.

Training and Evaluation
The training process involves feeding paired pre-processed images (filtered and grayscale) into the CNN model. The network is trained using sparse categorical cross-entropy as the loss function and evaluated on publicly available datasets - CEDAR and BHSig260. These contain both authentic signatures and skilled forgeries, providing diverse samples for robust model evaluation.

# Method 2:

The proposed offline signature verification is a combination of SURF algorithm and non-linear
SVM. This approach is writer independent.

Feature Extraction using SURF

The system performs feature extraction using the SURF algorithm. SURF is a local feature descriptor known for its speed and robustness. It approximates the determinant of the Hessian matrix to find keypoints in the signature images that capture crucial structural details such as stroke curves, intersections, and sharp turns. These keypoints are then described using SURF descriptors, which form a set of numerical features that characterize the local shape and texture of the signature.

Classification using Support Vector Machine (SVM)

SVM is a linear discriminant classifier used to verify signature images. It classifies signatures as either genuine or forged, particularly when a user ID is entered. SVM is favored for its high accuracy and relative simplicity. There are two main types of SVM: linear and non-linear. The method implemented separates classes linearly by identifying the optimal hyperplane. In the context of signature verification, the SVM is trained on a set of known genuine and forged signatures.The testing set of the images is then compared with the trained set to classify new signatures.

-------------------------------------------------------------------------------------------------------

# Preprocessing
# Method 1:
1.Image Loading: The image is read in grayscale mode using OpenCV (cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)).
2.Resizing: The image is resized to a fixed dimension of 256×128 pixels using OpenCV's cv2.resize. This ensures uniformity across all input images.
3.Filtering:
i.Dilation: The image undergoes dilation using a 3×3 kernel (cv2.dilate). This operation enhances features such as edges or strokes in the signature.
ii.Gaussian Blur: A Gaussian blur is applied with a 5×5 kernel (cv2.GaussianBlur). This smoothens the image and reduces noise.
iii.Thresholding: Otsu's thresholding is applied (cv2.threshold) to convert the image into a binary format (black and white). This is particularly useful for isolating signature strokes from the background.
4.Output: Two versions of the processed image are returned:
Filtered Image: The binary version after thresholding.
Grayscale Image: A copy of the original grayscale image.

# Method 2:
1. Binarization: This process transforms the color image to grayscale and then converts the
grayscale image to a binary image, leading to a clearer contour of the signature.
2. Noise Removal: A Gaussian filter has been used to extract noise and amplify signature image
structures, also blurring the images and reducing noise.
3. Resizing: This process involves resizing of all the photos to the same size (800 x 1500 pixels).
--------------------------------------------------------------------------------------------------------
