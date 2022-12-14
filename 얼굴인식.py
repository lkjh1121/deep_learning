#셀레니움은 크롬서 해야할 경우에 
from selenium import webdriver
from selenium.webdriver.common.by import By 
import time #시간 지연 
path = "./driver/chromedriver.exe"

from bs4 import BeautifulSoup 
from selenium.webdriver.common.keys import Keys 
import requests   #이미지를 다운로드 받을때 url.lib 둘 중하나  requests  최신임 
import time 
import os, shutil   #shutil - 디렉토리에 파일이 있거나 또는 다른 디렉토리가 있을때 싹 다지운다

def init(keyword): 
    driver = webdriver.Chrome(path) #크롬드라이버파일을 로딩한다 
    driver.implicitly_wait(3)#3초만 대기타라 
    driver.get("http://google.co.kr")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys( keyword ) #검색할 키워드 
    search_box.submit()

    return driver #객체 반환 

#페이지 다운로드  pgdn키를  누르면 된다. 이 신호를 드라이버한테 준다 
def pageDown(driver):
    #앞선페이지의 높이를  prevHeight에 저장해둔다  
    prevHeight = driver.execute_script("return document.body.scrollHeight")
    #마지막까지 스크롤되었는지 확인하기 위해 사용하는 변수 
    #화면내리기  -- window- 브라우저를 자바스크립트로 접근하는 객체
    # scrollTo - 화면 내리기 
    while True:
        #화면 내리고 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        current_Height = driver.execute_script("return document.body.scrollHeight")
        #현재화면의 높이 
        if prevHeight == current_Height: #앞선화면의 높이와 현재 높이가 같으면 더이상 
            break                        #내려갈데가 없으므로 while문 종료 
        prevHeight = current_Height  #앞선화면의 높이를 업데이트 한다 

def scrollEnd(driver, keyword): 
    #이미지 탭 선택하기 
    #hdtb-msb > div:nth-child(1) > div > div:nth-child(2) > a
    ele = driver.find_element(By.CSS_SELECTOR, "#hdtb-msb > div:nth-child(1) > div > div:nth-child(2) > a")
    ele.click()
    time.sleep(2)
    while True:
        pageDown(driver)
        try:
            time.sleep(2)
            ele2 = driver.find_element(By.CSS_SELECTOR, 
                "#islmp > div > div > div > div > div.gBPM8 > div.qvfT1 > div.YstHxe > input")
            ele2.click()
        except Exception as e:
            print(e)
            break #버튼이 더이상 없으면 이곳으로 온다. while문 종료 

import base64 #이미지태그에 이미지 데이터가 있는 경우에 base64를 이용해 저장하자 
def fileSave(driver, keyword):
    imgList = driver.find_elements(By.CSS_SELECTOR, "#islrg > div.islrc > div > a.wXeWr.islib.nfEiy > div.bRMDJf.islir > img")
    for i, img in enumerate(imgList):
        try:
            img.click()
            time.sleep(2)
            ele = driver.find_element(By.CSS_SELECTOR, "#Sva75c > div > div > div > div.pxAole > div.tvh9oe.BIB1wf > c-wiz > div.nIWXKc.JgfpDb.cZEg1e > div.OUZ5W > div.zjoqD > div.qdnLaf.isv-id.b0vFpe > div > a > img")
            time.sleep(2)
            src = ele.get_attribute('src')  
            if "data:image" not in src: #이미지 url 이 있는 경우에는 바로 다운받는다 
                img_res = requests.get(src)
                if img_res.status_code==200: #무사히 다운로드 되었을때 처리 
                    with open(f"./google/{keyword}/image_{i}.jpg", "wb") as fp:
                        fp.write(img_res.content)
            else:
                pos = src.index("/", 11) # 두번째 슬래시를 찾는다 
                data = src[:pos]
                #문자열을 -> base64형태로 디코딩작업을 해야 한다. 
                decoded_data =  base64.b64decode( data )    
                with open(f"./google/{keyword}/image_{i}.jpg", "wb") as fp:
                        fp.write(decoded_data)
                print(f"{i} saved")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    keyword = input("검색어 : ")
    path2 = "./google/"+keyword 
    if os.path.exists(path2): # 폴더가 존재하면 폴더 전체 삭제
        shutil.rmtree(path2, ignore_errors=True, onerror=None)
        os.mkdir(path2)
    else:
        os.mkdir(path2)

    driver = init(keyword)
    scrollEnd(driver, keyword)
    fileSave(driver, keyword)


