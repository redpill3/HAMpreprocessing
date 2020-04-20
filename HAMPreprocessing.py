#!/usr/bin/env python
# coding: utf-8

# ### Image sharpening 관련 함수들 선언 (3가지 필터중 어느 필터를 사용할건지가 변수)

# In[33]:


import cv2

def output(img, kernel_sharpen):
    #applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_sharpen)


    #displaying the difference in the input vs output
    #quits window if q is pressed
    #switches between the two images when any other key is pressed
    quit = False
    while(not quit):
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if(key == ord('q')):
            quit = True
            break;
        cv2.imshow('image', output)
        key = cv2.waitKey(0)
        if(key == ord('q')): #quit the window if q is pressed.
            quit = True
    #Destroys the open window
    cv2.destroyAllWindows()





def sharpen(path):
    #reading the image passed thorugh the command line
    img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    #process and output the image
    output(img, kernel)

def excessive(path):
    #reading the image
    img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])

    #process and output the image
    output(img, kernel)

def edge_enhance(path):
    #reading the image
    img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0

    #process and output the image
    output(img, kernel)


# In[39]:


outimg = sharpen('data/nv/ISIC_0024339.jpg')
outimg


# In[40]:


outimg = excessive('data/nv/ISIC_0024339.jpg')
outimg


# In[41]:


outimg = edge_enhance('data/nv/ISIC_0024339.jpg')
outimg


# ###  contrast 조절함수 (enhance 계수를 얼마나 줄것인가가 변수)

# In[24]:


from PIL import Image, ImageEnhance
inputimage = Image.open('data/nv/ISIC_0024339.jpg')
inputimage



# In[30]:


outimg = ImageEnhance.Contrast(image).enhance(1.3)
outimg


# In[ ]:




