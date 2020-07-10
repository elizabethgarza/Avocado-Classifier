import argparse
import imageio
import typing

from skimage.transform import resize

import preprocess
import train


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Trains image data.")
    parser.add_argument("num_iterations", help="path to directory that contains data")
    parser.add_argument("learning_rate", help="name of the fruit which you which to classify")
    parser.add_argument("directory_path", help="path to directory that contains data")
    parser.add_argument("fruit", help="name of the fruit which you which to classify")
    parser.add_argument("image_path", help="path to image that you'd like to classify")
    args = parser.parse_args()

    # prepares the data
    x_train, x_test, y_train, y_test = preprocess.prepare_data(args.directory_path, args.fruit)

    # trains the model
    model = train.model(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    num_iterations = int(args.num_iterations), 
    learning_rate = float(args.learning_rate), 
    print_cost = False
    )
 
    # loads and reads the image you want to classify
    im_pix = imageio.imread(args.image_path)
    im_pix = im_pix/255.
    im_pix = resize(im_pix, (100, 100),
                       anti_aliasing=True)
    im_pix = im_pix.reshape(100*100*3, 1)
    prediction = train.predict(model["w"], model["b"], im_pix)

    # prints out a prediction
    if str(prediction[0][0]) == '1.0':
        print(f"Prediction: {args.fruit}")
    
    if str(prediction[0][0]) == '0.0': 
        print(f"Prediction: Not a/an {args.fruit}")

