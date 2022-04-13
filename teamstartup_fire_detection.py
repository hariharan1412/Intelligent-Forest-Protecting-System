from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os
import cv2
from datetime import datetime
import smtplib
import imghdr
from email.message import EmailMessage
import winsound
from tkinter import *
from PIL import Image, ImageTk

class fire_detection:

    def __init__(self):
            
        self.cam = cv2.VideoCapture(0)
        self.img_counter = 0
        self.img = cv2.imread('1.jpg')
        self.execution_path = os.getcwd()

        self.mail_from = 'teamstartup.ht143@gmail.com'
        self.mail_password = 'godsense143'

        self.mgs = EmailMessage()
        self.mgs['subject'] = 'Forest Fire Detected! '
        self.mgs['From'] = self.mail_from
        self.mgs['To'] = 'teamstartuptesting@gmail.com'
        self.mgs.set_content('Forest Fire has been detected on camera 1 location ')

    def detect_from_image(self, img_name):
        self.detector = CustomObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(detection_model_path=os.path.join(self.execution_path, "models/teamStartup_fire_model.h5"))
        self.detector.setJsonPath(configuration_json=os.path.join(self.execution_path, "detection_config.json"))
        self.detector.loadModel()
    
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path, img_name),
                                                    output_image_path=os.path.join(self.execution_path, "1-detected.jpg"),
                                                    minimum_percentage_probability=40)

        currentTime = datetime.now()
        timeCurrent = currentTime.strftime("%d%m%Y%H%M%S")
            
        day = timeCurrent[0:2]
        month = timeCurrent[2:4]
        year = timeCurrent[4:8]
        hour = timeCurrent[8:10]
        min = timeCurrent[10:12]
        sec = timeCurrent[12:14]

        file_name = day +"d-"+ month +"m-"+ year +"y-"+ hour +"h-"+ min +"m-"+ sec+"s"
        file_name = "Fire Detected/Fire-"+file_name+".jpg"
        
        if len(detections):
            print("Fire Detected")     
            f = 2000
            d = 600
            winsound.Beep(f , d)
            os.replace("1-detected.jpg", file_name)
            self.alert(path=file_name)

        else:
            print("fire not detected")
            os.remove('1.jpg')

    def alert(self , path):

        path = path
        with open(path , 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name

        self.mgs.add_attachment(file_data,maintype = 'image',subtype = file_type, filename = file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
            smtp.login(self.mail_from,self.mail_password)
            smtp.send_message(self.mgs)
    
        image = cv2.imread(path)
        
        window_name = 'image'
        
        cv2.imshow(window_name, image)

    def main(self):

        a = 10
        counted = True
        start = True

        while start:            
            try:
                ret, frame = self.cam.read()
        
                font = cv2.FONT_HERSHEY_SIMPLEX
        
                currentTime = datetime.now()
                currentSecs = currentTime.strftime("%S")
        
                if counted:
                    a = int(currentSecs) + 10
                    c = int(currentSecs)
                    if a >= 60:
                        a -= 60
                    counted = False
        
                b = int(currentSecs) - c
                displaySecs = 10 - b
                if displaySecs >=60:
                    displaySecs = displaySecs - 60
                if displaySecs != 0:
        
                    cv2.putText(frame, 
                                'Capturing Image In ' + str(displaySecs), 
                                (100, 50), 
                                font, 1, 
                                (0,0,0), 
                                2, 
                                cv2.LINE_4)
                
                if displaySecs == 0:
                    cv2.putText(frame, 
                                'Image Captured', 
                                (100, 50), 
                                font, 1, 
                                (0,0,0), 
                                2, 
                                cv2.LINE_4)
                
                if not ret:
                    print("failed to grab frame")
                    break
        
                cv2.imshow("test", frame)
        
                currentTime = datetime.now()
                secondsNow = currentTime.strftime("%S")
        
                ms = currentTime.strftime("%f")
        
                k = cv2.waitKey(1)
        
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    start = False
        
                if a == int(currentSecs):
                    img_name = "1.jpg"
                    cv2.imwrite(img_name, frame)
                    self.detect_from_image(img_name)
                    counted = True
            
            except:
                pass
            self.cam.release()



