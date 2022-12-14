import cv2
import os, shutil

# src_path : 원본파일 있는곳
# dest_path : 목적지 파일 있는곳
def imageCut(src_path, dest_path):
    # 주의 : opencv 폴더명에 공백이나 한글있으면 오류 메세지를 출력
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    i=1
    for imgfilename in os.listdir(src_path):
        if os.path.isdir(imagefilename):
            continue 
        img = cv2.imread(src_path+"/"+imgfilename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴이 여러개를 다 검출한다.

        for (x, y, w, h) in faces:
            dst = img[y:y+h, x:x+y] # 이미지를 복사
            cv2.imwrite(f"{dest_path}/image{i}.jpg", dst)
            i = i+1
    cv2.destroyAllWindows() # 사용한 메모리 os에게 반환
imageCut("./csy", "./csy/cut_image")







# 쓸데없음