This project classifies images into categories using Support Vector Machine (SVM) with PCA for feature reduction.

Note: I have uploaded some ZIP files of images which are private and not accessible to others.
To test or classify images, please upload your own image folders or ZIP files.

Steps:
1. Extract ZIP image files.
2. Preprocess images and extract HOG features.
3. Apply PCA and train an SVM model using GridSearchCV.
4. Evaluate performance using accuracy, classification report, and confusion matrix.

Requirements:
pip install numpy opencv-python matplotlib seaborn scikit-learn scikit-image


Run the script:
python Index.py


Output:
Accuracy score
Classification report
Confusion matrix heatmap
