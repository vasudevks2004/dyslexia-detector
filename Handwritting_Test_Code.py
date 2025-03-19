import cv2
import numpy as np
from scipy import ndimage as nd
from scipy import ndimage
import joblib
import pressure
import zones
import segmentation
from skimage.feature import graycomatrix, graycoprops
from tkinter import filedialog

def compute_feats(image, kernels):
    """Extracts Gabor texture features."""
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def GLCM_Feature(cropped):
    """Extracts GLCM features from the cropped signature."""
    glcm = graycomatrix(cropped, [1, 2], [0, np.pi/2], levels=256, normed=True, symmetric=True)
    features = ['dissimilarity', 'correlation', 'energy', 'contrast', 'homogeneity', 'ASM']
    
    glcm_features = np.hstack([graycoprops(glcm, prop).flatten() for prop in features])
    return glcm_features

# Personality Classes
list1 = ['strong personality', 'moderate personality', 'weak personality']

# Read Image
S_filename = filedialog.askopenfilename(title='Select Signature Image')

if not S_filename:
    print("No file selected. Exiting.")
    exit()

S_img = cv2.imread(S_filename)

if S_img is None:
    print("Error loading image. Please select a valid image file.")
    exit()

# Convert to Grayscale
G_img = cv2.cvtColor(S_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input Image', G_img)
cv2.waitKey(0)

# Gaussian Filtering & Thresholding
blurred_image = ndimage.gaussian_filter(G_img, sigma=2)
_, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Segmented Image', binarized_image)
cv2.waitKey(0)

# Find Center of Mass & Crop
r, c = np.where(binarized_image == 0)
if r.size == 0 or c.size == 0:
    print("Error: No signature detected in the image.")
    exit()

cropped = G_img[r.min(): r.max(), c.min(): c.max()]

# Feature Extraction
Average, Percentage = pressure.pressure(cropped)
top, middle, bottom = zones.findZone(cropped)
Glcm_feature_signature = GLCM_Feature(cropped)

bw_img, angle1 = segmentation.Segmentation(G_img)

# Combine Features
feature_matrix1 = np.concatenate((Average, Percentage, angle1, top, middle, bottom, Glcm_feature_signature), axis=None)
feature_matrix1 = feature_matrix1.reshape(1, -1)  # Ensure correct shape for model input

# Load Model & Predict
try:
    Model_lod1 = joblib.load("Trained_H_Model.pkl")
    pred = Model_lod1.predict(feature_matrix1)
    print("Prediction:", list1[int(pred[0])])
except FileNotFoundError:
    print("Error: Model file 'Trained_H_Model.pkl' not found. Train and save the model first.")
except Exception as e:
    print(f"Prediction Error: {e}")

