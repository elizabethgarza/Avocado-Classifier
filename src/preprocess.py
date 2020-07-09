import argparse
import numpy as np
import os
import pandas as pd
import sys

from PIL import Image
from typing import List


Vector = List[List]

def get_true_data(directory: str, fruit: str) -> Vector:
    
    """
    This function retrieves image pixel data for all images that you wish to classify as "True" and puts it into an 3-D array.
    
    Arguments:   
    directory -- The name of the directory in which you have downloaded 'fruits-360'.
    fruit -- The name of the fruit that you wish to classify.  The name of this fruit must match the name of a subdirectory within 'fruits-360'.  

    Returns:
    true_im_pixels -- A 3-D numpy array of size (m, 100*100, 3) containing images of the fruits you wish to classify as "True".
    """
    
    directory = directory + "/fruits-360/Test"
    
    # iterates through directory to retrieve pixel values for true_images and appends those values to a 2-D array
    true_im_pixels = np.empty((10000,3), int)

    sub_dirs = os.listdir(directory)
    if fruit not in sub_dirs:
        print(f"Error: the fruit {fruit} does not match a sub-directory. fruit must be one of {str(sub_dirs)}")
        sys.exit(0)

    for sub_directory in os.listdir(directory): 
        if sub_directory == fruit: 
            start_path = directory + f"/{fruit}"
            # im is the name of the image, which is also a subdirectory
            for im in os.listdir(start_path):
                if im == ".DS_Store": 
                    continue 
                else:
                    end_path = start_path + f"/{im}/fruits-360/Test/{fruit}/{im}"
                    im = Image.open(end_path, 'r')
                    width, height = im.size
                    im_pixels = np.array(im.getdata())
                    true_im_pixels = np.append(true_im_pixels, im_pixels, axis=0)
                
    # reshapes all_im_pixels from 2-D to 3-D.
    m = int(true_im_pixels.shape[0]/10000)
    true_im_pixels = true_im_pixels.reshape(m, 100*100, 3)
    
    return true_im_pixels


def get_false_data(directory: str, fruit: str) -> Vector: 
        
    """
    This function retrieves image pixel data for all images that will be labeled with a "1"--
    i.e. true data--and puts it into an 3-D array.
    
    Arguments:   
    directory -- The name of the directory in which you have downloaded 'fruits-360'.
    fruit -- The name of the fruit that you wish to classify.  The name of this fruit must match the name of a subdirectory within 'fruits-360'. 
      
    Returns:
    true_im_pixels -- A 3-D numpy array of size (m, 100*100, 3) containing random images of all fruits in `fruits-360`, except the fruit which you which to classify.
    """
        
    directory = directory + "/fruits-360/Test"

    false_im_pixels = np.empty((10000,3), int)
    for sub_directory in os.listdir(directory): 
        if sub_directory.startswith(fruit):
            continue
        if ".DS_Store" in sub_directory: 
            continue
        else:
            start_path = directory + f"/{sub_directory}"
            # extracts only one image file from each directory
            i=0
            for im in os.listdir(start_path):
                i+=1
                if i>1:
                    continue
                end_path = start_path + f"/{im}/fruits-360/Test/{sub_directory}/{im}"
                im = Image.open(end_path, 'r')
                width, height = im.size
                im_pixels = np.array(im.getdata())
                false_im_pixels = np.append(false_im_pixels, im_pixels, axis=0)
                    
    m = int(false_im_pixels.shape[0]/10000)
    false_im_pixels = false_im_pixels.reshape(m, 100*100, 3)
    
    return false_im_pixels

def prepare_data(directory_path, fruit):

    # instantiates true_im_pixels and false_im_pixels objects
    true_im_pixels = get_true_data(directory_path, fruit)
    false_im_pixels = get_false_data(directory_path, fruit)
    
    # labels both arrays with y values
    true_im_pix_and_tags = np.ones((true_im_pixels.shape[0], true_im_pixels.shape[1], true_im_pixels.shape[2]+1))
    true_im_pix_and_tags[:, :, :-1] = true_im_pixels
    
    false_im_pix_and_tags = np.zeros((false_im_pixels.shape[0], false_im_pixels.shape[1], false_im_pixels.shape[2]+1))
    false_im_pix_and_tags[:, :, :-1]= false_im_pixels
    
    # concatenates and shuffles the arrays
    all_pix_and_tags = np.concatenate((true_im_pix_and_tags, false_im_pix_and_tags))
    np.random.shuffle(all_pix_and_tags)
    
    # ensures that each image element in the array has 4 columns
    assert(all_pix_and_tags.shape[2]==4)

    # stores the y-labels into a (1, 3) array
    y = np.empty((1, all_pix_and_tags.shape[0]), dtype=np.int64)
    i = 0
    for array in all_pix_and_tags: 
        y[0][i] = array[0][3]
        i+=1

    # splits y-labels into y_train and y_test
    m_train = round(y.shape[-1]*.8)
    m_test = y.shape[-1] - m_train
    
    y_train = np.empty((1, m_train), int)
    i = 0
    for element in y[0][0:m_train]:
        y_train[0][i] = element
        i+=1
    
    # ensures that the number of training labels is equal to the number of m training examples
    assert(y_train.shape[1]==m_train)

    y_test = np.empty((1, m_test), int)
    i=0
    for element in y[0][m_train:]: 
        y_test[0][i] = element
        i+=1

    # ensures that the number of test labels is equal to the number of m test examples
    assert(y_test.shape[1]==m_test)

    # prints y_train and y_test to a csv file for user's reference, if needed.
    df_y_train = pd.DataFrame(y_train)
    df_y_train.to_csv('y_train.csv')

    df_y_test = pd.DataFrame(y_test)
    df_y_test.to_csv('y_test.csv')

    # deletes the labels from apple_pix_and_tags
    x_start = np.delete(all_pix_and_tags[0], 3, 1)
    for array in all_pix_and_tags[1:]: 
        x = np.vstack((x_start, (np.delete(array, 3, 1))))
        x_start = x

    # flattens the 3-D array to a printable 2-D array of (m, 100*100*3) dimensions
    x = x.reshape(int(x.shape[0]/10000), 100*100*3)

    # ensures that the number of x image arrays is equal to the number of y labels
    assert((round(y.shape[-1]*.8)) == (round(x.shape[0]*.8)))

    # splits into x arrays into train and test sets
    x_train = x[:(round(y.shape[-1]*.8))]
    x_test = x[(round(y.shape[-1]*.8)):]

    # flattens and standardizes x_train and x_test
    x_train_flatten = x_train.reshape(x_train.shape[1], x_train.shape[0])
    x_test_flatten = x_test.reshape(x_test.shape[1], x_test.shape[0])

    x_train_stan = x_train_flatten/255
    x_test_stan = x_test_flatten/255

    return x_train_stan, x_test_stan, y_train, y_test
   

if __name__=="__main__": 
    
    parser = argparse.ArgumentParser(description="Preprocess image data.")
    parser.add_argument("directory_path", help="path to directory that contains data")
    parser.add_argument("fruit", help="name of the fruit which you which to classify")
    args = parser.parse_args()

    prepare_data(args.directory_path, args.fruit)


