:brain: Brain Tumor Detection using Image Processing
________________________________________________________

* Frameworks 📒 :
  
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

* Libraries 📖 :

![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

* Operating Systems 🖥️ :
  
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)

* Platform 💻 :

![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)


1. Summary of the Project 🖋️ :
* This project aims to develop a system for detecting brain tumors using image processing techniques.
* The system processes MRI scans to identify the presence of tumors, leveraging various image processing algorithms and machine learning models to achieve high accuracy.

2. Introduction 🗒️ :
* Brain tumor detection is a critical task in medical imaging that can significantly impact patient outcomes.
* This project utilizes advanced image processing and machine learning techniques to automatically detect tumors from MRI scans, providing a valuable tool for medical professionals.

3. Features 📚 :
* Preprocessing of MRI images (resizing, normalization, etc.)
* Tumor segmentation using various image processing techniques
* Feature extraction from segmented regions
* Classification of MRI scans as tumor or non-tumor
* Visualization of detected tumors

4. Usage 🛠️ :
* To detect brain tumors using this system, start by preprocessing the MRI data.
* This involves resizing, normalizing, and augmenting the images, which can be done by running the preprocess.py script with the appropriate input and output directories specified: python preprocess.py with input directory set to path/to/mri/images and output directory set to path/to/preprocessed/images.
* Once the data is preprocessed, proceed to train the model by running the train.py script, specifying the directory containing the preprocessed images and the directory where the trained model should be saved: python train.py with data directory set to path/to/preprocessed/images and model directory set to path/to/save/model.
* Finally, to detect tumors in new MRI scans, use the trained model by running the detect.py script with the model directory, the path to the new MRI image, and the path to save the output image: python detect.py with model directory set to path/to/saved/model, input image set to path/to/new/mri/image, and output image set to path/to/save/detected/image.
* This sequence of steps enables the system to preprocess data, train a model, and detect tumors in new images.

5. Dataset 📖 :
* The dataset used for this project consists of MRI scans with labeled tumor regions.
* Sample Dataset source (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

6. Model Training ⚡ :
* Data Augmentation: Enhance the training dataset with various transformations to improve model robustness.
* Model Architecture: Define a convolutional neural network (CNN) architecture suitable for image classification.
* Training: Train the CNN model on the training dataset, optimizing it to accurately distinguish between tumor and non-tumor images.
* Validation: Validate the model using a separate validation set to tune hyperparameters and prevent overfitting.

6. Model Evaluation ⚙️ :
* Evaluate the trained model using the test dataset to assess its performance.
* Key metrics include accuracy, precision, recall, and F1-score.
* Visualize the results with confusion matrices to understand the model's strengths and weaknesses.

7. Results and Implementation 📊 :
* The Model achieved 99.19% Accuracy 📈
* The Model achieved 0.03% Loss 📉


