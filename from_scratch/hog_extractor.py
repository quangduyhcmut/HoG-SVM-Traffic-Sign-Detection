import os
import sys
import shutil
import cv2

def load_img(root_dir = './test_img/1', img_name = ['1.png']):
    arr = []
    for name in img_name:
        img = cv2.imread(os.path.join(root_dir, name))
        arr.append(img)
    return arr

def imshow_img(img_list, break_index):
    for index, img in enumerate(img_list):
        cv2.imshow("original image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if index == break_index:
            break
        
def get_size(img_list):
    size_list = []
    shape_list = []
    for img in img_list:
        size_list.append(img.size)
        shape_list.append(img.shape)
    return size_list, shape_list

def get_hog_skimage(img_list):
    hog_list = []
    for img in img_list:
        pass

def main():
    
    root_dir = '../test_img/1'
    img_name = ['38.png']
    # img_name = os.listdir(root_dir)
    img_loaded = load_img(root_dir, img_name)
    
    # # imshow some first image:
    # imshow_img(img_loaded, break_index =2)
    
    # # check image size:
    # print("Number of traffic signs", len(img_loaded))
    size_list, shape_list = get_size(img_loaded)
    # print("Max area: ", max(size_list)/3)
    # print("Min area: ", min(size_list)/3)
    # print(shape_list)
    
if __name__ == "__main__":
    main()