import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# 사용할 모델과 설정 값 정의
model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
min_confidence = 0.3
file_name = "/Users/skytm/OpenCV/image/marathon_01.jpg"
title_name = 'dnn Deep Learnig object detection'
frame_width = 300
frame_height = 300

# 파일 선택 함수
def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)

    # 선택한 이미지 파일 읽기 및 Tkinter용 이미지로 변환
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    
    # 이미지 크기 정보 및 객체 검출 함수 호출
    (height, width) = read_image.shape[:2]
    detectAndDisplay(read_image, width, height)

# 얼굴 검출 및 화면에 출력하는 함수 정의
def detectAndDisplay(frame, w, h):
    # Caffe 모델 로드
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # 입력 이미지를 300x300 픽셀로 조정하고 정규화한 후 모델에 전달
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    # 최소 신뢰도 설정
    min_confidence = float(sizeSpin.get())
    print(min_confidence)

    # 검출 결과 반복
    for i in range(0, detections.shape[2]):
        
            # 예측에 관련된 신뢰도(확률) 추출
            confidence = detections[0, 0, i, 2]

            # 최소 신뢰도보다 큰 경우에만 처리
            if confidence > min_confidence:
                
                    # 객체의 경계 상자 좌표 계산
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    print(confidence, startX, startY, endX, endY)
     
                    # 얼굴 주변에 경계 상자 그리기 및 신뢰도 표시
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 결과 이미지 출력
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image = imgtk
    
#main
main = Tk()
main.title(title_name)
main.geometry()

# 이미지 초기 설정 및 Tkinter PhotoImage로 변환
read_image = cv2.imread(file_name)
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)
(height, width) = read_image.shape[:2]

# Tkinter GUI 구성요소 및 레이아웃 설정
label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)
sizeLabel=Label(main, text='Min Confidence : ')                
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)

# 초기 이미지로 객체 검출 함수 호출
detectAndDisplay(read_image, width, height)

main.mainloop()
