import os
import zipfile
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from sklearn.decomposition import PCA

zips = [
    r"C:\Users\Admin\OneDrive\Desktop\Diseases-Detection-Project\Grade-1.zip",
    r"C:\Users\Admin\OneDrive\Desktop\Diseases-Detection-Project\Grade-2.zip",
    r"C:\Users\Admin\OneDrive\Desktop\Diseases-Detection-Project\Grade-3.zip"
]
out_dir = r"C:\Users\Admin\OneDrive\Desktop\Diseases-Detection-Project\extracted_grades"
Path(out_dir).mkdir(parents=True, exist_ok=True)

for z in zips:
    if os.path.exists(z):
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(out_dir)

def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.equalizeHist(img)
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

X, y = [], []
for grade in sorted(os.listdir(out_dir)):
    folder = os.path.join(out_dir, grade)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(preprocess_image(img))
            y.append(grade)

X, y = np.array(X), np.array(y)
if len(X) == 0:
    raise RuntimeError("No images found after extraction.")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=300, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {"C": [1, 10, 50], "gamma": ["scale", 0.01], "kernel": ["rbf"]}
grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=1)
grid.fit(X_train_pca, y_train)
model = grid.best_estimator_

y_pred = model.predict(X_test_pca)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

labels = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
