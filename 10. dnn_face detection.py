import cv2
import numpy as np

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
min_confidence = 0.3
file_name = "/Users/skytm/OpenCV/image/marathon_01.jpg"

# 얼굴 검출 및 화면 출력 함수 정의
def detectAndDisplay(frame):
    # Caffe 모델 로드
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # 입력 이미지 300x300 픽셀로 조정 후 정규화한 후 모델에 전달
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    # 각 얼굴 검출 결과에 대해 반복
    for i in range(0, detections.shape[2]):
            # 예측에 관련된 신뢰도(확률) 추출
            confidence = detections[0, 0, i, 2]

            # 신뢰도가 최소 신뢰도보다 큰 경우에만 처리
            if confidence > min_confidence:
                
                    # 객체의 경계 상자 좌표 계산
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
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
    cv2.imshow("Face Detection by dnn", frame)


print("OpenCV version:")
print(cv2.__version__)
 
img = cv2.imread(file_name)
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
