import os
import shutil
import random
import cv2 as cv
import numpy as np

def pre_process(src):
    img = cv.resize(src, (640, 480))
    img_gaus = cv.GaussianBlur(img, (5,5), 0)

    img_B = cv.split(img_gaus)[0]
    img_G = cv.split(img_gaus)[1]
    img_R = cv.split(img_gaus)[2]

    img_gray = cv.cvtColor(img_gaus, cv.COLOR_BGR2GRAY)
    img_hsv = cv.cvtColor(img_gaus, cv.COLOR_BGR2HSV)

    return img, img_gaus, img_B, img_G, img_R, img_gray, img_hsv

def raw_ID(img_gray, img_hsv, img_B, img_R):
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if (abs(img_hsv[:,:,0][i,j]-115) < 15) and (img_B[i,j] > 70) and (img_R[i,j] < 40):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0

    kernel1 = np.ones((3, 3))
    kernel2 = np.ones((7, 7))

    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_dilate = cv.dilate(img_gray, kernel1, iterations=4)
    img_close = cv.morphologyEx(img_dilate, cv.MORPH_CLOSE, kernel2)
    img_close = cv.GaussianBlur(img_close, (5, 5), 0)
    _, img_bin = cv.threshold(img_close, 100, 255, cv.THRESH_BINARY)
    return img_bin

def pos_detection(img, img_bin):
    _, contours,_ = cv.findContours(img_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    det_x_max = 0
    det_y_max = 0
    pts = 0
    for i in range(len(contours)):
        x_min = np.min(contours[i][ :, :, 0])
        y_min = np.min(contours[i][ :, :, 1])
        x_max = np.max(contours[i][ :, :, 0])
        y_max = np.max(contours[i][ :, :, 1])
        det_x = x_max - x_min
        det_y = y_max - y_min
        if (det_x / det_y > 1.8) and (det_x > det_x_max) and (det_y > det_y_max):
            det_x_max = det_x
            det_y_max = det_y
            pts = i
    points = np.array(contours[pts][:, 0])
    return points

def find_vertices(points):
    rect = cv.minAreaRect(points)
    #  print(rect)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]])

    return vertices, rect

def distortion_correction(vertices, rect):
    if rect[2] > -45:
        new_right_point_x = vertices[0, 0]
        new_right_point_y = int(vertices[1, 1] - (vertices[0, 0]- vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (vertices[1, 1] - vertices[3, 1]))
        new_left_point_x = vertices[1, 0]
        new_left_point_y = int(vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (vertices[2, 1] - vertices[0, 1]))
        pts1 = np.float32([[440, 0],[0, 0],[0, 140],[440, 140]])
    elif rect[2] < -45:
        new_right_point_x = vertices[1, 0]
        new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (vertices[1, 1] - vertices[2, 1]))
        pts1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])

    new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]), (new_right_point_x, new_right_point_y)])
    pts0 = np.float32(new_box)
    return pts0, pts1, new_box

def transform_license(img, pts0, pts1):
    mat = cv.getPerspectiveTransform(pts0, pts1)
    license = cv.warpPerspective(img, mat, (440, 140))
    return license

if __name__ == "__main__":
    dir_name = "./raw_imgs/"
    target_dir_name = "./dataset/"
    file_names = os.listdir(dir_name)
    random.shuffle(file_names)

    for file_name in file_names:
        print("\n{}".format(file_name))
        img = cv.imread(dir_name+file_name)
        cv.imshow("image", img)
        img, img_gaus, img_B, img_G, img_R, img_gray, img_hsv = pre_process(img)
        img_bin = raw_ID(img_gray, img_hsv, img_B, img_R)
        points = pos_detection(img, img_bin)
        vertices, rect = find_vertices(points)
        pts0, pts1, new_box = distortion_correction(vertices, rect)
        img_draw = cv.drawContours(img.copy(), [new_box], -1, (0,0,255), 3)
        cv.imshow("con", img_draw)
        license = transform_license(img, pts0, pts1)
        cv.imwrite(target_dir_name+file_name, license)
        cv.imshow("license", license)
        cv.waitKey(0)

        cv.destroyAllWindows()
