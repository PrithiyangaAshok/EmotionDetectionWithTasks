import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"
#TASK
img_size = 48

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json","model.weights.h5")


EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def process_video_stream():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame = cv2.resize(frame, (800, 600))  # Resize for display
        processed_frame = detect_emotion(frame)
        
        cv2.imshow('Emotion Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_emotion(frame):        
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = facec.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (img_size, img_size))
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Predict emotion using the loaded model
            preds = model.predict(roi)
            emotion_label = EMOTIONS_LIST[np.argmax(preds)]

            # Draw bounding box and label on the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            return frame

# TASK 2 Starting
#Dynamic model adaptation: Develop a system that adapts its emotion recognition model based on individual user data or real-time feedback, leading to personalized predictions.

def feedback_callback(emotion_index):
            
            # Perform actions based on user feedback
            print("User feedback received for emotion:", EMOTIONS_LIST[emotion_index])
            # Here, you can update the model based on the user's feedback

        
feedback_button = Button(top, text="Feedback", padx=10, pady=5)
feedback_button.configure(
            background="#364156", foreground="white", font=("arial", 10, "bold"),
            command=lambda: feedback_callback(pred_idx)
        )
feedback_button.place(relx=0.79, rely=0.56) 

#TASK 2 COMPLETION



def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def upload_image():
    file_path = filedialog.askopenfilename()
    try:
        uploaded_image = Image.open(file_path)
        uploaded_image.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded_image)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
    except Exception as e:
        print(f"Error: {e}")

upload_button = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_button.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

# Start real-time video processing
process_video_stream()

top.mainloop()