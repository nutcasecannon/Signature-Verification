Offline Signature Verification Using CNN inspired by InceptionV1
Reference Research Paper : DOI:10.1109/ICIIP47207.2019.8985925

The code is run on Kaggle using TPU for faster computation.

Datasets Used : CEDAR, BHSig-260 (Hindi and Bengali)
https://www.kaggle.com/datasets/ankita22053139/cedarbhsig-260

-------------------------------------------------------------------------------------------------------
Dataset Description

CEDAR
The dataset utilized for this offline signature verification study is the CEDAR signature dataset, a widely recognized benchmark in signature verification research. This dataset comprises a total of 2,640 images, collected from 55 persons. Each individual contributed 24 genuine signatures, amounting to 1,320 authentic samples, along with 24 forged signatures, leading to an equal number of 1,320 forged samples. All images are in .png format. The forged signatures were carefully created to mimic the genuine ones, making the dataset highly suitable for evaluating the robustness of signature verification systems.

BHSig260 signature dataset
The BHSig260 dataset is composed of two subsets: BHsig-B (Bengali) and BHsig-H (Hindi). The BHSig260 dataset is used for research in offline signature verification, specifically for multilingual scenarios.
§BHSig260 (Bengali) signature dataset contains signatures from 100 persons each having 24 authentic and 30 forged signatures, totaling 2,400 authentic and 3,000 forged signatures.
§BHSig260 (Hindi) signature dataset contains signatures from 160 persons each having 24 authentic and 30 forged signatures, totaling 3,840 authentic and 4,800 forged signatures.

-------------------------------------------------------------------------------------------------------
Methodology

This methodology for offline handwritten signature verification utilizes a Convolutional Neural Network (CNN) inspired by the Inception V1 (GoogleNet) architecture. The model adopts a writer-independent approach, treating genuine and forged signatures as distinct classes.

InceptionSVGNet Architecture
The architecture of InceptionSVGNet builds upon the principles of Inception V1, emphasizing parallel convolutional layers with varying kernel sizes to extract multi-scale features effectively. The model incorporates multiple filters at each layer level, enhancing feature diversity without significantly increasing depth.

1.Inception Modules: Each Inception module consists of four parallel convolutional layers with kernel sizes of 16×16, 8×8, 4×4, and 2×2. These layers operate simultaneously on the input data, extracting features at different scales. To create a single feature map, the outputs from these layers are concatenated along the channel dimension.
2.Layer Configuration: The network begins with an initial convolutional block that generates 64 feature maps from the input tensor. Three successive Inception modules follow, with increasing filter counts (16, 24, and 32 filters in each module) and decreasing spatial dimensions through average pooling layers. A final convolutional layer with 16 filters further refines the feature representation before flattening.
3.Classification : After flattening the feature maps, two connected dense layers are used for high-level feature learning. Dropout regularization (rate = 0.5) is applied between these layers to prevent overfitting. Finally, a softmax layer outputs probabilities for the two classes: genuine and forged signatures.

Training and Evaluation
The training process involves feeding paired pre-processed images (filtered and grayscale) into the CNN model. The network is trained using sparse categorical cross-entropy as the loss function and evaluated on publicly available datasets - CEDAR and BHSig260. These contain both authentic signatures and skilled forgeries, providing diverse samples for robust model evaluation.

-------------------------------------------------------------------------------------------------------

Preprocessing
1.Image Loading: The image is read in grayscale mode using OpenCV (cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)).
2.Resizing: The image is resized to a fixed dimension of 256×128 pixels using OpenCV's cv2.resize. This ensures uniformity across all input images.
3.Filtering:
i.Dilation: The image undergoes dilation using a 3×3 kernel (cv2.dilate). This operation enhances features such as edges or strokes in the signature.
ii.Gaussian Blur: A Gaussian blur is applied with a 5×5 kernel (cv2.GaussianBlur). This smoothens the image and reduces noise.
iii.Thresholding: Otsu's thresholding is applied (cv2.threshold) to convert the image into a binary format (black and white). This is particularly useful for isolating signature strokes from the background.
4.Output: Two versions of the processed image are returned:
Filtered Image: The binary version after thresholding.
Grayscale Image: A copy of the original grayscale image.
--------------------------------------------------------------------------------------------------------
