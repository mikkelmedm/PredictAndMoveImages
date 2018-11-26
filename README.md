# PredictAndMoveImages

Move images to new folder wrt. the retrained MobileNet class they belong to. 
Can also be used for other Keras models. Simply load in your own model (this one works specifically for MobileNet) and run it. 

Works great for large image folders, where specific images should be kept for training on GAN models and others not.

Simply run the RearrangeImages.py file, and a new directory will added be separating the images into to new folders for each class. The program expects your images to be in a folder called "images", however another input folder can be specified by running the program with the --folder_dir flag. example:
```
RearrangeImages.py --folder_dir myfolder
```
