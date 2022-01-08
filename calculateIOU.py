import cmath
import numpy as np
import os

#calculate IOU calue for a fram
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, w1, h1]
    :param box2: = [xmin2, ymin2, w2, h2]
    :return:
    """
    xmin1, ymin1, w1, h1 = box1
    xmin2, ymin2, w2, h2 = box2

    s1 = w1 * h1  # b1 area
    s2 = w2 * h2  # b2 area

    # intersection area
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmin1+w1, xmin2+w2)
    ymax = min(ymin1+h1, ymin2+h2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G area
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou

# calculate CLE value
def cal_cle(box1, box2):
    p1 = [box1[0]+box1[2]*0.5, box1[1]+box1[3]*0.5]
    p2 = [box2[0]+box2[2]*0.5, box2[1]+box2[3]*0.5]

    distance = abs(p1[0]-p2[0])*abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])*abs(p1[1]-p2[1])
    distance = cmath.sqrt(distance)
    return distance

# read data from file
video_name = 'test_01'
img_path1 = 'D:/TrackingWildlife_project/TestVideo'
gts_path = os.path.join(img_path1, video_name, 'groundtruth.txt')
result_path = os.path.join(img_path1, video_name, 'result.txt')
fps_path = os.path.join(img_path1, video_name, 'fps.txt')
result = np.loadtxt(result_path,delimiter=',')
gts = np.loadtxt(gts_path, delimiter=',')
IOU = np.zeros(len(result))
CLE = np.zeros(len(result))

#do calculation
for i in range(len(result)):
    IOU[i] = cal_iou(gts[i],result[i])
    CLE[i] = cal_cle(gts[i],result[i])
    print(IOU[i],'CLE',CLE[i],'\n')

meanIOU = np.mean(IOU)
meanCLE = np.mean(CLE)

#show result
print('IOU',meanIOU,'\nCLE',meanCLE)
