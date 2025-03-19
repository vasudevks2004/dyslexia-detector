from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
##
import glob
import json
import h5py
import imutils
from sklearn.datasets import load_files
from keras.models import load_model
import time
##
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm       
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
##
from scipy import ndimage as nd
from scipy import ndimage
import joblib
import pressure
import zones
import feature_extraction
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import confusion_matrix
##
from imutils.video import VideoStream
#
import segmentation
from tkinter import filedialog
##
import speech_recognition as sr
import string
import threading

def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        # changing the title of our master widget      
        self.master.title("DYSLEXIA DETECTOR")
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        w = tk.Label(root, 
		 text=" Dyslexia Prediction using Handwritting, Eye Movement and reading test",
		 fg = "light blue",
		 bg = "white",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=50, y=10)
        # creating a button instance
        quitButton = Button(self,command=self.Train_writting,text="Train Handwritting Image",fg="blue",activebackground="dark red",width=20)
        # placing the button on my window
        quitButton.place(x=50, y=60)
        quitButton = Button(self,command=self.Train_Eye_Movement,text="Train Eye movement Image",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=250, y=60)
        quitButton = Button(self,command=self.EYE_Tracking,text="Start Eye Tracking",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=450, y=60)
        quitButton = Button(self,command=self.Handwriting_Prediction,text="Select Handwritting Image",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=650, y=60)
        quitButton = Button(self,command=self.start_speech_test_thread,text="start speech test",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=850, y=60)

        load = Image.open("logo.png")
        logo_img = ImageTk.PhotoImage(load)     
        image1=Label(self, image=logo_img,borderwidth=2, highlightthickness=5, height=300, width=400, bg='white')
        image1.image = logo_img
        image1.place(x=50, y=120)

        self.T = Text(self, height=20, width=43)
        self.T.pack()
        self.T.place(x=650, y=100)
        self.T.insert(END, "Waiting for Results...")


    def Train_writting(self, event=None):
        self.T.delete("1.0", tk.END)  # Clear existing text
        self.T.insert(tk.END, "Handwriting Feature Extraction and Training...")
        print("Handwriting Feature Extraction and Training...")
  
        CNN_DATA1=[]
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
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, "Handwriting Feature Extraction and Training Completed.")
        print("Handwriting Feature Extraction and Training Completed.")

        
    def Train_Eye_Movement(self, event=None):
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, "Training Eye Movement...")
        print("Training Eye Movement...")

        
        data2=[]
        data=[]
        featurematrix=[]
        label=[]
        label2=[]
        cw_directory = os.getcwd()
        #cw_directory='D:/Hand gesture/final_code'
        folder=cw_directory+'/eye dataset'
        for filename in os.listdir(folder):
            
            sub_dir=(folder+'/' +filename)
            for img_name in os.listdir(sub_dir):
                img_dir=str(sub_dir+ '/' +img_name)
                print(int(filename),img_dir)
                img = cv2.imread(img_dir)
                # Resize image
                img = cv2.resize(img,(128,128))
                if len(img.shape)==3:
                    img2 = cv2.resize(img,(32,32))
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img2=img2.flatten()
                    data2.append(img2/255.0)
                    label2.append(int(filename))
                    
                data11=np.array(img)
                data.append(data11/255.0)
                label.append(int(filename))
         

        #target1=train_targets[label]
        ##

        def train_CNN(data,label):
            ##
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))

            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(36))

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
           
            X_train, Y_test, X_label, Y_label = train_test_split(data,label, test_size=0.20)

            history = model.fit(np.array(X_train), np.array(X_label), epochs=20, 
                                validation_data=(np.array(Y_test), (Y_label)))
            
            show_history_graph(history)
            test_loss, test_acc = model.evaluate(np.array(Y_test), np.array(Y_label), verbose=2)
            print("Testing Accuracy is ", test_acc)
            print("Testing loss is ", test_loss)

            #hist=model.fit(np.array(data), (train_targets) ,validation_split=0.1, epochs=10, batch_size=64)
            model.save('eye_movement_trained.h5')
            return model

        # CNN Training
        model_CNN = train_CNN(data,label)
        Y_CNN=model_CNN.predict(np.array(data))
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, "Training Eye Movement Completed.")
        print("Training Eye Movement Completed.")


    def EYE_Tracking(self, event=None):
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, "Starting Eye Movement based Dyslexia Prediction...")
        print("Starting Eye Movement based Dyslexia Prediction...")


        list1 = ['looking at center', 'looking at left', 'looking at right', 'looking at up', 'looking at down']

        # Load trained CNN model
        model_path = 'trained_model_CNN1.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")

        eye_cnn = tf.keras.models.load_model(model_path)

        # Histogram equalization function
        def histogram_equalization(img):
            if img is None:
                raise ValueError("Error: Image not loaded properly.")
    
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            r, g, b = cv2.split(img)
            f_img1 = cv2.equalizeHist(r)
            f_img2 = cv2.equalizeHist(g)
            f_img3 = cv2.equalizeHist(b)
            img = cv2.merge((f_img1, f_img2, f_img3))
            return img

        # Function to get index positions
        def get_index_positions_2(list_of_elems, element):
            return [i for i, val in enumerate(list_of_elems) if val == element]

        # Define model paths
        base_dir = os.path.dirname(__file__)
        prototxt_path = os.path.join(base_dir, 'model_data/deploy.prototxt')
        caffemodel_path = os.path.join(base_dir, 'model_data/weights.caffemodel')

        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            raise FileNotFoundError("Caffe model files not found! Check 'deploy.prototxt' and 'weights.caffemodel'.")

        # Load face detection model
        model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Load eye cascade
        eye_cascade_path = 'haar cascade files/haarcascade_eye.xml'
        if not os.path.exists(eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade file {eye_cascade_path} not found!")

        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        # Start video capture
        vs = cv2.VideoCapture(0)
        eyemovement = []
        Dyslexia_result = []
        n1, n2 = 0, 10

        while True:
            ret, frame = vs.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame = imutils.resize(frame, width=750, height=512)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            model.setInput(blob)
            detections = model.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.40:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                f_img = frame[startY:endY, startX:endX]
                if f_img.shape[0] == 0 or f_img.shape[1] == 0:
                    continue

                f_img = histogram_equalization(f_img)
                roi_gray = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(roi_gray)

                pred = None
                for cn, (ex, ey, ew, eh) in enumerate(eyes):
                    if cn == 1:
                        one_eye = cv2.resize(f_img[ey:ey+eh, ex:ex+ew], (28, 28))  # Resize to (28,28)
                        one_eye = cv2.cvtColor(one_eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                        one_eye = one_eye.astype('float32') / 255.0  # Normalize pixel values
                        one_eye = np.expand_dims(one_eye, axis=-1)  # Add channel dimension (28,28,1)
                        one_eye = np.expand_dims(one_eye, axis=0)  # Add batch dimension (1,28,28,1)

                        # Predict eye movement
                        predictions = eye_cnn.predict(one_eye)
                        pred = np.argmax(predictions, axis=-1)[0]  # Extract the predicted index

                        # Debugging: Print model output
                        print(f"Raw predictions: {predictions}")
                        print(f"Predicted index: {pred}")

                        eyemovement.append(pred)

                    cv2.rectangle(f_img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    frame[startY:endY, startX:endX] = f_img
                    if cn == 1:
                        break

                # Ensure `pred` is within valid range
                if pred is None or pred >= len(list1) or pred < 0:
                    print(f"Warning: Invalid prediction index {pred}, defaulting to 'Eyes Not Detected'.")
                    text = "Eyes Not Detected"
                else:
                    text = list1[pred]

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 200, 200), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 150), 2)

            cv2.imshow("Frame", frame)

            # Dyslexia detection logic
            if len(eyemovement) >= 10 and len(eyemovement) >= n2:
                eye_array = eyemovement[n1:n2]
                Dyslexia_result.append(1 if len(np.unique(eye_array)) > 2 else 0)
                n1 += 20
                n2 += 20

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vs.release()
        cv2.destroyAllWindows()

        # Count occurrences of positive and negative results
        number_of_positive = get_index_positions_2(Dyslexia_result, 1)
        number_of_negative = get_index_positions_2(Dyslexia_result, 0)

        if len(number_of_positive) >= 10 or len(number_of_positive) > len(number_of_negative):
            print("Symptoms of Dyslexia detected")
            contents = "Symptoms of Dyslexia detected"
            Dyslexia_result = 1
        else:
            print("Symptoms of Dyslexia NOT detected")
            contents = "Symptoms of Dyslexia NOT detected"
            Dyslexia_result = 0

        # Update Tkinter text box
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, contents)
        self.Dyslexia_eye = Dyslexia_result

        
    def Handwriting_Prediction(self, event=None):
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, "Starting Handwriting-based Dyslexia Prediction...")
        print("Starting Handwriting-based Dyslexia Prediction...")

        def compute_feats(image, kernels):
            feats = np.zeros((len(kernels), 2), dtype=np.double)
            for k, kernel in enumerate(kernels):
                filtered = nd.convolve(image, kernel, mode='wrap')
                feats[k, 0] = filtered.mean()
                feats[k, 1] = filtered.var()
            return feats

        def GLCM_Feature(cropped):
            # GLCM Feature extraction
            glcm = graycomatrix(cropped, [1, 2], [0, np.pi/2], levels=256, normed=True, symmetric=True)
            dissim = (graycoprops(glcm, 'dissimilarity'))
            dissim=np.reshape(dissim, dissim.size)
            correl = (graycoprops(glcm, 'correlation'))
            correl=np.reshape(correl,correl.size)
            energy = (graycoprops(glcm, 'energy'))
            energy=np.reshape(energy,energy.size)
            contrast= (graycoprops(glcm, 'contrast'))
            contrast= np.reshape(contrast,contrast.size)
            homogen= (graycoprops(glcm, 'homogeneity'))
            homogen = np.reshape(homogen,homogen.size)
            asm =(graycoprops(glcm, 'ASM'))
            asm = np.reshape(asm,asm.size)
            glcm = glcm.flatten()
            Mn=sum(glcm)
            Glcm_feature = np.concatenate((dissim,correl,energy,contrast,homogen,asm,Mn),axis=None)
            return Glcm_feature

        list1= ['Dyslexia Handwriting', 'Normal Handwriting']

            #Read Image
        S_filename = filedialog.askopenfilename(title='Select Signature Image')
        S_img = cv2.imread(S_filename)
        Sh_img=cv2.resize(S_img,(300,50))
        cv2.imwrite('H_Image.png',Sh_img)        
        if len(S_img.shape) == 3:
            G_img = cv2.cvtColor(S_img, cv2.COLOR_RGB2GRAY)
        else:
            G_img=S_img.copy()

        load = Image.open("H_Image.png")
        logo_img = ImageTk.PhotoImage(load)     
        image1=Label(self, image=logo_img,borderwidth=2, highlightthickness=5, height=300, width=400, bg='white')
        image1.image = logo_img
        image1.place(x=50, y=120)
        
        cv2.imshow('Input Image',cv2.resize(G_img,(300,50)))
        cv2.waitKey(0)           
            #Gaussian Filter and thresholding image
        blur_radius = 2
        blurred_image = ndimage.gaussian_filter(G_img, blur_radius)
        threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Segmented Image',cv2.resize(binarized_image,(300,50)))
        cv2.waitKey(0)
            # Find the center of mass
        r, c = np.where(binarized_image == 0)
        r_center = int(r.mean() - r.min())
        c_center = int(c.mean() - c.min())

            # Crop the image with a tight box
        cropped = G_img[r.min(): r.max(), c.min(): c.max()]

            ## Signature Feature extraction
        Average,Percentage = pressure.pressure(cropped)
        top, middle, bottom = zones.findZone(cropped)

        Glcm_feature_signature =GLCM_Feature(cropped)
        Glcm_feature_signature=Glcm_feature_signature.flatten()

        bw_img,angle1= segmentation.Segmentation(G_img)

        feature_matrix1 = np.concatenate((Average,Percentage,angle1,top, middle, bottom,Glcm_feature_signature),axis=None)

        Model_lod1 = joblib.load("Trained_H_Model.pkl")

        #ypred = Model_lod.predict(cv2.transpose(Feature_matrix))
        pred=Model_lod1.predict(cv2.transpose(feature_matrix1))
        Dyslexia_writing=pred[0]
        print(pred)
        contents=list1[pred[0]]
        self.T.delete("1.0", tk.END)
        self.T.insert(tk.END, contents)
        print(contents)

        self.Dyslexia_writing=Dyslexia_writing

    def start_speech_test_thread(self):
        threading.Thread(target=self.start_speech_test, daemon=True).start()

    def start_speech_test(self):
        paragraph = "Artificial intelligence (AI) is intelligence demonstrated by machines."
        
        # Display paragraph before starting speech recognition
        self.T.delete("1.0", tk.END)
        self.T.insert(END, f"Please read the paragraph aloud:\n\n{paragraph}\n\n")
        self.T.update()  # Force immediate update
        time.sleep(1)  # Give the UI time to refresh

        detected_text = self.listen_to_paragraph()

        if detected_text:
            self.T.insert(END, f"\nDetected Speech: {detected_text}\n")
            self.check_for_dyslexia(paragraph, detected_text)

    def listen_to_paragraph(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.T.insert(END, "Listening...\n")
            self.T.update()
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            self.T.insert(END, "\nCould not understand the audio.\n")
        except sr.RequestError as e:
            self.T.insert(END, f"\nError with the speech recognition service: {e}\n")
        return None

    def check_for_dyslexia(self, original, detected):
        original_words = original.lower().translate(str.maketrans('', '', string.punctuation)).split()
        detected_words = detected.lower().translate(str.maketrans('', '', string.punctuation)).split()
        missing_words = [word for word in original_words if word not in detected_words]

        self.T.insert(END, f"\nMissing words: {missing_words}\n")

        if len(missing_words) >= 2:
            self.T.insert(END, "You may have dyslexia. Please consider consulting a professional.\n")
        else:
            self.T.insert(END, "You read the paragraph well!\n")

root = Tk()
#size of the window
root.geometry("1050x450")
app = Window(root)
root.mainloop()  
