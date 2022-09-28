import numpy as np
def convolution3x3(krenel , img : np.array ):
    result_img = np.zeros([img.shape[0]-2,img.shape[1]-2])
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            result_img[i][j] = np.sum(img[i:i+3,j:j+3] * krenel)
    return result_img
def convolution5x5(krenel , img : np.array ):
    result_img = np.zeros([img.shape[0]-4,img.shape[1]-4])
    for i in range(img.shape[0]-4):
        for j in range(img.shape[1]-4):
            result_img[i][j] = np.sum(img[i:i+5,j:j+5] * krenel)
    return result_img