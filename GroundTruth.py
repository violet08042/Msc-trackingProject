import cv2 as cv
import os


drawing = False
ix, iy = -1, -1
w, h = -1, -1


def draw_rect(event,x,y,flags,param):
    global ix, iy, drawing, w, h
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = False
        cv.rectangle(img,(ix,iy),(x,y),(0,255,0),1,4)
        w = x - ix
        h = y - iy


img_path = 'D:/tracking_related/TestVideo/test_01'
cv.namedWindow('get ground truth')
f = open(os.path.join(img_path, 'groundtruth.txt'),'w')
i = 0
FrameNumber = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]) - 1
Key = cv.waitKey(0)
while True:
    if Key & 0xFF == 27:
        break

    if Key & 0xFF == 13:
        if i >= FrameNumber:
            break
        img_name = 'test_01_' + str(i + 1000) + '.jpg'
        img = cv.imread(os.path.join(img_path, img_name))
        cv.setMouseCallback('get ground truth', draw_rect)
        cv.imshow('get ground truth', img)
        i = i + 1
        cv.waitKey(0)
        print(ix, ',', iy, ',', w, ',', h)
        f.write(str(ix)+','+str(iy)+','+str(w)+','+str(h)+'\n')
f.close()
cv.destroyAllWindows()


