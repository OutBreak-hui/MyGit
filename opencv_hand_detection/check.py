import cv2
import numpy as np
import time

import math


def cal_whites(img):
    # img=cv2.imread(photo)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = img.shape[1]
    height = img.shape[0]

    total = width * height
    whites = []
    for x in range(height):
        for y in range(width):
            if img[x, y][0] == 255:
                whites.append(1)

    print('whites像素比例---', len(whites) / total)
    return (len(whites) / total)


def cal_hsv(origin, mask, single_contour):
    poly = cv2.fillPoly(mask, [single_contour], (255, 0, 0))
    height = poly.shape[0]
    width = poly.shape[1]
    hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
    H = []
    S = []
    V = []
    for x in range(width):
        for y in range(height):
            if poly[y, x][0] == 255:
                H.append(hsv[y, x][0])
                S.append(hsv[y, x][1])
                V.append(hsv[y, x][2])
    meanH = np.mean(H)
    meanS = np.mean(S)
    meanV = np.mean(V)
    print('=================')
    print('H----', meanH)
    print('S----', meanS)
    print('V----', meanV)
    return meanH, meanS, meanV


def ellipse_detect(image):
    img = image
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (118, 155), (25, 15), 43, 0, 360, (255, 255, 255), -1)
    # cv2.ellipse(skinCrCbHist ,(113,155),(23,15),43,0, 360, (255,255,255),-1)##标准

    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255

    dst = cv2.bitwise_and(img, img, mask=skin)

    # bgr=cv2.cvtColor(dst,cv2.COLOR_YCR_CB2BGR)
    bgr = cv2.medianBlur(dst, 5)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                3)  ##自适应阈值，通过取峰值来自
    cv2.imshow("before", bgr)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 2000:
            #         # bin=cv2.drawContours(image, [cnt], -1, (0,0,255), 2)
            bin = cv2.fillPoly(bgr, [cnt], (0, 0, 0))

    bin = cv2.drawContours(bgr, contours, -1, (0, 0, 255), 1)
    cv2.imshow("bin_cutout", bgr)


def skin_by_otsu(image):  ##YCrCb颜色空间的Cr分量

    img = image
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dst = cv2.bitwise_and(img, img, mask=skin)  ##分离出的人的肌肤区域

    # cv2.imshow("seperate",dst)
    bgr = cv2.medianBlur(dst, 3)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin = cv2.dilate(gray, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)==1:
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            cv2.fillPoly(bgr, [cnt], (0, 0, 0))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bin_cut = 0
    bgr_cut = 0
    bgr_cut_tag = 0

    bin = cv2.dilate(bin, kernel, iterations=5)
    bin = cv2.erode(bin, kernel, iterations=5)

    new_cnts = []
    areas = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        if perimeter > 500 and area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            # print("长除以高",w/h)
            new_cnts.append(cnt)
            areas.append(area)

            # hull = cv2.convexHull(cnt)
            # cv2.drawContours(bgr, [hull], -1, (0,0,255), 3)

    if len(areas) <= 1:
        pass
    else:
        MAX = areas.index(max(areas))
        cv2.fillPoly(bin, [new_cnts[MAX]], (0, 0, 0))
        cv2.fillPoly(bgr, [new_cnts[MAX]], (0, 0, 0))

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 3000:
            # x,y,w,h = cv2.boundingRect(cnt)
            # print("w/h-----",w/h)
            # print('area---',area)
            # if w/h<0.75:
            #     my_cnt=[[[int(x),int(y+h/2)]],[[int(x+w),int(y+h/2)]],[[int(x+w),int(y+h)]],[[[int(x),int(y+h)]]]]

            #     cv2.fillPoly(bin,my_cnt,(0,0,0))

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    kernel = np.ones((5, 5), np.uint8)
    # bin=cv2.erode(bin,kernel,iterations = 20)
    # bin=cv2.dilate(bin,kernel,iterations = 2)
    # bin=cv2.erode(bin,kernel,iterations = 3)
    # bin=cv2.dilate(bin,kernel,iterations = 2)
    # bin=cv2.erode(bin,kernel,iterations = 3)
    # bin=cv2.dilate(bin,kernel,iterations = 5)

    # if len(cont)==1:
    #     x,y,w,h = cv2.boundingRect(cont[0])
    #     cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)

    # for cn in cont:
    #     are = cv2.contourArea(cn)
    #     if are>600:
    #         rect = cv2.minAreaRect(cn)
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)
    #         bgr=cv2.drawContours(bgr,[box],0,(0,0,255),2)
    # hull = cv2.convexHull(cn)
    # cv2.drawContours(bgr, [hull], -1, (0,0,255), 3)

    cv2.imshow('bgr', img)
    cv2.imshow('bin', bin)

    # perimeter= cv2.arcLength(cnt,True)
    # area = cv2.contourArea(cnt)
    # if perimeter>500 and area>800:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     print('meiyou sdfa  sd')

    #         # bin_cut=skin[y:y+h,x:x+w]
    #         # bgr_cut=bgr[y:y+h,x:x+w]

    # cv2.imshow("bgr",bgr)

    # kernel = np.ones((5,5),np.uint8)
    # cv2.dilate(bin_cut,kernel,iterations = 5)
    # cv2.erode(bin_cut,kernel,iterations = 30)

    #         # bgr_cut_tag=1000
    #         print('white-----',cal_whites(bin))
    #         if cal_whites(bin_cut)>0.35:##多次膨胀之后白色部分比例改变让然很小,就是脸部形状
    #             pass
    #         else:
    #             cv2.dilate(bin_cut,kernel,iterations = 5)
    #             contours, hierarchy = cv2.findContours(bin_cut,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #             for cnt in contours:
    #                 x,y,w,h = cv2.boundingRect(cnt)
    #                 cv2.rectangle(bgr_cut,(x,y),(x+w,y+h),(0,255,0),2)
    #                 bgr_cut_tag=1000

    #         # cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)
    #         # cv2.drawContours(bgr, [cnt], -1, (0,0,255), 3)

    #     else:
    #         cv2.fillPoly(dst,[cnt],(0,0,0))

    # if bgr_cut_tag==1000:
    #     cv2.imshow('cut',bgr_cut)
    #     cv2.imshow('bin_cut',bin_cut)

    # bin=bin_cut

    # bin=cv2.Canny(bin_cut,0,400)
    #         count.append(area)
    # print('轮廓数量',len(count))

    # MAX=count.index(max(count))
    # poly=cv2.fillPoly(bgr,[contours[MAX]],(255,0,0))

    # if len(count)==1:##只有一个轮廓只有三种情况，只有人脸，一只手和脸在一起，两只手和脸在一起
    # for cnt in contours:
    #     perimeter = cv2.arcLength(cnt,True)

    #     area = cv2.contourArea(cnt)

    #     if perimeter>800 and area>1500:
    #         count.append(area)

    #         x,y,w,h = cv2.boundingRect(cnt)
    #         cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)
    #         cv2.fillPoly(bgr,[cnt],(255,0,0))
    #         # cv2.drawContours(bgr, [cnt], -1, (0,0,255), 3)

    #     bin=bgr

    # ret, bin= cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # bin=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 3) ##自适应阈值，通过取峰值来自

    # contours, hierarchy = cv2.findContours(bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # for cnt in contours:

    #     area = cv2.contourArea(cnt)
    #     if area<100:
    # #         # bin=cv2.drawContours(bgr, [cnt], -1, (0,255,0), 2)
    #         bin=cv2.fillPoly(bin,[cnt],(0,0,0))
    # #     else:
    # #         bin=cv2.fillPoly(bgr,[cnt],(0,0,255))

    # bin=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3) ##自适应阈值，通过取峰值来自
    # _,bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5,5),np.uint8)
    # bin = cv2.dilate(bin,kernel,iterations = 3)
    # bin = cv2.erode(bin,kernel,iterations = 20)

    # contours, hierarchy = cv2.findContours(bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # for cnt in contours:

    #     area = cv2.contourArea(cnt)
    #     if area>2000:
    #         x,y,w,h = cv2.boundingRect(cnt)
    #         bin = cv2.rectangle(bin,(x,y),(x+w,y+h),(0,0,255),2)
    #         bin=cv2.drawContours(bin, [cnt], -1, (0,255,0), 2)
    # bin=cv2.fillPoly(dst,contours,(255,255,255))

    # cv2.imshow("after",bin)


c = 1
capture = cv2.VideoCapture(0)
timeF = 20  # 视频帧计数间隔频率
while (True):
    # 获取一帧
    ret, frame = capture.read()
    skin_by_otsu(frame)
    cv2.waitKey(3)

    # ellipse_detect(frame)
    # if(c%timeF == 0): #每隔timeF帧进行存储操作
    #     skin_by_otsu(frame)

    # if cv2.waitKey(1) == ord('q'):
    #     break
    # c = c + 1