import argparse
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import os, sys
import cv2
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras
import shutil

# This "with CustomObjectScope" is specific for MobileNet:
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    # Load your Keras model here:
    model = load_model('RetrainedMobileNet.h5')
# To run other Keras models, simply load the model with load_model without the CustomObjectScope

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder_dir', type=str, default='images',
                    help='folder with images to split up')

args = parser.parse_args()


print("I'm working...")

path = args.folder_dir
# Specify path-name for new directory containing the images you want to move.
newpath = "Newimgdir"
dirs = os.listdir( path )

if not os.path.exists(newpath):
    os.makedirs(newpath)

# Names of new folders: 
goodfolder = newpath+"/"+"good" 
badfolder = newpath+"/"+"bad" 

#Creating new folders:
if not os.path.exists(goodfolder):
    os.makedirs(goodfolder)
if not os.path.exists(badfolder):
    os.makedirs(badfolder)

def predictandmove():
    count1=0
    count0=0
    for item in dirs:
        if os.path.isfile(path+"/"+item):
            img_path=path+"/"+item
            img = cv2.resize(cv2.imread(img_path), (224, 224)) # 224,224 is specific for this model.
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x) 

            labels = np.argmax(preds, axis=-1)
            
            # if a given class, move to a give folder:
            if labels==1:
                shutil.move(path+"/"+item, goodfolder)
                count1+=1
            elif labels==0:
                shutil.move(path+"/"+item, badfolder)
                count0+=1

    print("")
    print("-----------------------")
    print("Finished!")
    print("")
    print("Moved",count0,"bad images to",badfolder)
    print("and",count1,"good images to",goodfolder)
    print("")


predictandmove()
