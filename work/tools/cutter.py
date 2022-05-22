import os.path
import shutil

import cv2
import numpy as np
import random
from PIL import Image

# 最大类间方差法
def otsu(img):
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def adaptive_threshold(gray, blockSize=2, C=10, inv=False):
    if inv == False:
        thresholdType = cv2.THRESH_BINARY
    else:
        thresholdType = cv2.THRESH_BINARY_INV
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType, blockSize, C)
    return binary_img


def get_projection_list_demo(binary_img):
    h, w = binary_img.shape[:2]
    row_list = [0] * h
    col_list = [0] * w
    for row in range(h):
        for col in range(w):
            if binary_img[row, col] == 0:
                row_list[row] = row_list[row] + 1
                col_list[col] = col_list[col] + 1

    temp_img_1 = 255 - np.zeros((binary_img.shape[0], max(row_list)))
    for row in range(h):
        for i in range(row_list[row]):
            temp_img_1[row, i] = 0
    cv2.imshow('horizontal', temp_img_1)

    temp_img_2 = 255 - np.zeros((max(col_list), binary_img.shape[1]))
    for col in range(w):
        for i in range(col_list[col]):
            temp_img_2[i, col] = 0
    cv2.imshow('vertical', temp_img_2)


def get_projection_list(binary_img, direction='horizontal'):
    h, w = binary_img.shape[:2]
    row_list = [0] * h
    col_list = [0] * w
    for row in range(h):
        for col in range(w):
            if binary_img[row, col] == 0:
                row_list[row] = row_list[row] + 1
                col_list[col] = col_list[col] + 1
    if direction == 'horizontal':
        return row_list
    else:
        return col_list


def split_projection_list(projectionList: list, minValue=0):
    start = 0
    end = None

    split_list = []
    for idx, value in enumerate(projectionList):
        if value > minValue:
            end = idx
        else:
            if end is not None:
                split_list.append((start, end))
                end = None
            start = idx
    else:
        if end is not None:
            split_list.append((start, end))
            end = None
    return split_list


def cut_binary_img(binary_img, startX, startY, limit, direction='horizontal', iteration=2):
    img_h, img_w = binary_img.shape[:2]
    if iteration <= 0:
        return {
            'rect': (startX, startY, img_w, img_h),
            'childern': None
        }

    children = []

    projection_list = get_projection_list(binary_img, direction)
    minValue = int(0.1 * sum(projection_list) / len(projection_list))
    # print(minValue)
    split_list = split_projection_list(projection_list, minValue)
    for start, end in split_list:
        if end - start < limit:
            continue
        if direction == 'horizontal':
            x, y, w, h = 0, start, img_w, end - start
        else:
            x, y, w, h = start, 0, end - start, img_h

        roi = binary_img[y:y + h, x:x + w]
        if direction == 'horizontal':
            next_direction = 'vertical'
        else:
            next_direction = 'horizontal'
        grandchildren = cut_binary_img(roi, startX + x, startY + y, limit, next_direction, iteration - 1)

        children.append(grandchildren)

    root = {
        'rect': (startX, startY, img_w, img_h),
        'childern': children
    }
    return root


def get_leaf_node(root):
    leaf_rects = []
    if root['childern'] is None:
        leaf_rect = root['rect']
        leaf_rects.append(leaf_rect)
    else:
        for childern in root['childern']:
            rects = get_leaf_node(childern)
            leaf_rects.extend(rects)
    return leaf_rects


def draw_rects(img, rects):
    new_img = img.copy()
    for x, y, w, h in rects:
        p1 = (x, y)
        p2 = (x + w, y + h)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = (0, 0, 255)
        cv2.rectangle(new_img, p1, p2, color, 2)
    return new_img

def draw_rects_with_txt(img, txt):
    image = cv2.imread(img)
    f = open(txt)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        p1 = (int(line.split(',')[0]), int(line.split(',')[1]))
        p2 = (int(line.split(',')[4]), int(line.split(',')[5]))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(image, p1, p2, color, 2)
        line = f.readline()
    f.close()
    return image

def check_dir(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(os.path.join(save_path, "intermediate"))
    os.makedirs(os.path.join(save_path, "result"))


def line_cutter(path, img_name, point_coords, save_path, txt):
    image = cv2.imread(os.path.join(path, img_name))
    max_x, max_y = np.amax(point_coords, axis=0)
    min_x, min_y = np.amin(point_coords, axis=0)
    tmp = image[min_y:max_y, min_x: max_x]
    limit = min(max_x - min_x, max_y - min_y)/5

    ########################################
    vec = [-min_x, -min_y]
    revec = [min_x, min_y]
    pts = vec + point_coords
    pts = np.array([pts])
    # 和原始图像一样大小的0矩阵，作为mask
    mask = np.zeros((tmp.shape[:2]), np.uint8)
    # 在mask上将多边形区域填充为白色
    cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
    cv2.fillPoly(mask, pts, 255)  # 填充
    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst = cv2.bitwise_and(tmp, tmp, mask=mask)
    # 改白色背景
    bg = np.ones_like(tmp, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
    dst = bg + dst
    img = dst

    ##############################################
    # blurred = cv2.medianBlur(img, 5)
    # blurred = cv2.GaussianBlur(img, (3, 3), 0)
    blurred = cv2.pyrMeanShiftFiltering(img, 10, 50)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # binary_img = adaptive_threshold(gray, blockSize=15)
    binary_img = otsu(gray)
    binary_img = cv2.medianBlur(binary_img, 5)
    # binary_img = threshold(gray)

    # 再改白色背景
    bg = np.ones((tmp.shape[:2]), np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255

    binary_img = bg + binary_img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    erode_img = cv2.erode(binary_img, kernel)
    # erode_img = binary_img

    H = 'horizontal'
    V = 'vertical'
    root = cut_binary_img(erode_img, 0, 0, limit, direction=V, iteration=3)

    rects = get_leaf_node(root)

    for each in rects:
        char = image[each[1] + min_y: each[1] + min_y + each[3],
               each[0] + min_x: each[0] + min_x + each[2]]
        num = str(each[0] + min_x) + str(each[1] + min_y)
        name = os.path.join(save_path, 'intermediate', num + '.png')
        cv2.imwrite(name, char)


        x1 = each[0] + min_x
        y1 = each[1] + min_y
        x2 = each[0] + min_x + each[2]
        y2 = y1
        x3 = x2
        y3 = each[1] + min_y + each[3]
        x4 = x1
        y4 = y3
        string = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) \
                 + ',' + str(x3) + ',' + str(y3) + ',' + str(x4) + ',' + str(y4) \
                 + ',' + str(0) + ',' + 'text' + '\n'
        txt.write(string)





