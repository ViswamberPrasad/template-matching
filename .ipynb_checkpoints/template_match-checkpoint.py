import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

img = cv2.imread('ronaldo_foot.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

temp = cv2.imread('ronaldo_face.jpg')
temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
plt.imshow(temp)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    
    img_copy = img.copy()
    
    method = eval(m)
    
    res = cv2.matchTemplate(img_copy,temp,method)
    
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    height,width,channels = temp.shape
    
    cv2.rectangle(img_copy,top_left,(top_left[0]+width,top_left[1]+height),(0,0,255),15)
    
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP')
    plt.subplot(122)
    plt.imshow(img_copy)
    plt.title('TEMPLATE MATCHING')
    plt.suptitle(m)
    plt.show()
    
    print('\n')
    print('\n')