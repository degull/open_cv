import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# Haar_cascade 파일 경로 / 이미지 파일 경로
face_cascade_name = '/Users/skytm/OpenCV/data/haarcascade_frontalface_alt.xml'
eyes_cascade_name = '/Users/skytm/OpenCV/data/haarcascade_eye_tree_eyeglasses.xml'
file_name = '/Users/skytm/OpenCV/image/marathon_01.jpg'

# 윈도우 타이틀 설정
title_name = 'Haar cascaed object detection'
frame_width = 500

# 파일 불러오기 선택 함수
def selectFile():
    # 파일 대화상자 통해 이미지 파일 선택하도록
    file_name = filedialog.askopenfilename(initialdir = "./image", title = "Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print("File name : ", file_name)

    # 선택한 이미지 파일 읽고 크기 조정
    read_image = cv2.imread(file_name)
    (height, weight) = read_image.shape[:2]
    frameSize = int(sizeSpin.get())
    ratio = frameSize / width
    dimension = (frameSize, int(height* ratio))
    read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA)

    # 이미지를 RGB 형식으로 변환해  Tkinter용 PhotoImage로 변환
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)

    # 객체 검출 함수 호출
    detectAndDisplay(read_image)
    
# 얼굴 및 눈 검출 함수 
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

        # 눈 검출
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            
        #cv2.imshow('Capture - Face detection', frame)

        # Tkinter Label에 이미지 업데이트
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        detection.config(image=imgtk)
        detection.image = imgtk

# 메인 윈도우 생성
main = Tk()
main.title(title_name)
main.geometry()

# 이미지 초기 설
read_image = cv2.imread("/Users/skytm/OpenCV/image/marathon_01.jpg")
(height, width) = read_image.shape[:2]
ratio = frame_width / width
dimension = (frame_width, int(height * ratio))
read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA)

# OpenCV 이미지를 Tkinter PhotoImage로 변환
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

face_cascade_name = '/Users/skytm/OpenCV/data/haarcascade_frontalface_alt.xml'
eyes_cascade_name = '/Users/skytm/OpenCV/data/haarcascade_eye_tree_eyeglasses.xml'

# 얼굴 및 눈 검출을 위한 하르 캐스케이드 분류기 로딩
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()


#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

# Tkinter GUI 구성요소 및 레이아웃 설정
label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

sizeLabel=Label(main, text='Frame Width : ')                
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=frame_width)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=2000, increment=100, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))

detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)


# 초기 이미지로 객체 검출 함수 호출
detectAndDisplay(read_image)

# Tkinter 이벤트 루프 시작
main.mainloop()
