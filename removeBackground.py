import cv2
import numpy as np
import os

def removeBackground(img, filePath):
    src = img
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    print(filePath)
    cv2.imshow("dsa", dst)
    cv2.imwrite(filePath, dst)
    


for picture in os.listdir("./emojis"):
    path_to_img = "./emojis/"+picture
    if ".jpg" not in path_to_img and ".png" not in path_to_img:
        continue
    img = cv2.imread(path_to_img)
    removeBackground(img, "./emoji/" + picture)
