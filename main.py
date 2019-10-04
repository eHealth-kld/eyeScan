#import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from pymongo import MongoClient
from win10toast import ToastNotifier
import time
import dlib
import cv2

blinc_time = False


client = MongoClient('45.67.57.68:27017')
db = client.hd
collection = db.vision1
ses = db.session


toaster = ToastNotifier()

toaster.show_toast("Привет",
                   "Я прослежу за твоим здоровьем",
                   icon_path="ico/hd.ico",
                   duration = 5)

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def getBlancCount(count):
    time.sleep(30)
    return count*2

EYE_BLINC_GATE = 0.30 #Порог закрытия глаз
EYE_BLINC_FRAMES = 3
COUNTER = 0
TOTAL = 0

before_time = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("cascade/predictor.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

faceCascade = cv2.CascadeClassifier('cascades/face.xml')

print(str((lStart, lEnd)))

vs = cv2.VideoCapture(0)

fileStream = True


while True:
    
    ret, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    rects = detector(gray, 0)
    
    for rect in rects:
        
        start_time = time.time()
        
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
           
        if ear < EYE_BLINC_GATE: 
            COUNTER+=1
        else:
            if COUNTER >= EYE_BLINC_FRAMES:
                TOTAL+=1
                if (blinc_time != False):
                    result = time.time() - blinc_time  
                    data = {
                       "time" : result,
                        }
                    insert = collection.insert_one(data).inserted_id
                blinc_time = time.time()
            COUNTER = 0
            
        if (time.time() > 150000):
                toaster.show_toast("Внимание",
                   "Стоит отдохнуть, превышено допустимое время пребывания за компьютером",
                   icon_path="ico/hd.ico",
                   duration = 5)
            
                data = {
                    "check" : "1",
                    "text" : "Слишком долго за компьютером",
                    }
                insert = ses.insert_one(data).inserted_id
    
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
        


