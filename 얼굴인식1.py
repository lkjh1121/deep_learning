import cv2
import os
import sys


# # 주의 : opencv 폴더명에 공백이나 한글이 있으면 오류 메세지 출력
# img1 = cv2.imread("./google/csy/image_0.jpg")
# cv2.imshow("image1", img1)
# cv2.waitKey(0)          # 키 누를때까지 이미지를 화면에 출력하고 있다.
# cv2.destroyAllWindows() # 사용한 메모리 os에게 반환

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


img = cv2.imread("./google/csy/image_0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image1", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows() # 사용한 메모리 os에게 반환

faces = face_cascade.detecMultiScale(gray, 1.3, 5) # 얼굴이 여러개를 다 검출한다.
print(faces)


for (x, y, w, h) in faces:
    print(x, y, w, h ) # 얼굴 시작 위치 x 값 y w h
    # 영역에 선그리기
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("image1", img)
    cv2.waitKey(0)
cv2.destroyAllWindows() # 사용한 메모리 os에게 반환
