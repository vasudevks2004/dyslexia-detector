import cv2
import os
import numpy as np
from scipy import ndimage
import joblib
import pressure
import zones
import feature_extraction
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Initialize Data Lists
S_Data = []
S_label = []

# Get Dataset Path
cw_directory = os.getcwd()
H_dataset = os.path.join(cw_directory, 'Dataset2')

# Load Dataset & Extract Features
cnt = 0
for filename in os.listdir(H_dataset):
    sub_dir = os.path.join(H_dataset, filename)
    for img_name in os.listdir(sub_dir):
        img_dir = os.path.join(sub_dir, img_name)
        print(f"Processing: {img_dir}")
        
        # Ensure feature_extraction module has Feature_extraction function
        feature_matrix1 = feature_extraction.Feature_extraction(img_dir)
        
        S_Data.append(feature_matrix1)
        S_label.append(int(filename))  # Ensure filename represents a valid label
    cnt += 1
    print(f"Processed {cnt} classes")

# Split Data for Training & Testing
X_train, X_test, y_train, y_test = train_test_split(S_Data, S_label, test_size=0.2, random_state=42)

# Train ANN (MLP Classifier)
model1 = MLPClassifier(activation='relu', verbose=True, hidden_layer_sizes=(100,), batch_size=30)
model1.fit(np.array(X_train), np.array(y_train))
ypred_MLP = model1.predict(np.array(X_test))

# Display Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model1, np.array(X_test), np.array(y_test))
plt.show()

# ANN Accuracy
S_ACC = accuracy_score(y_test, ypred_MLP)
print("Testing ANN Accuracy:", S_ACC)

# Save Model
joblib.dump(model1, "Trained_H_Model.pkl")

# Train SVM Model
def train_SVM(featuremat, label):
    clf = SVC(kernel='rbf', random_state=0)
    clf.fit(np.array(featuremat), np.array(label))
    y_pred = clf.predict(np.array(featuremat))
    
    # Display Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(clf, np.array(featuremat), np.array(label))
    plt.show()
    
    print("SVM Accuracy:", accuracy_score(label, y_pred))
    return clf

svc_model1 = train_SVM(X_train, y_train)
Y_SCM_S_pred = svc_model1.predict(X_test)
SVM_S_ACC = accuracy_score(y_test, Y_SCM_S_pred)

# Plot ANN vs SVM Accuracy
plt.figure()
plt.bar(['ANN'], [S_ACC], label="ANN Accuracy", color='r')
plt.bar(['SVM'], [SVM_S_ACC], label="SVM Accuracy", color='g')
plt.legend()
plt.ylabel('Accuracy')
plt.show()
