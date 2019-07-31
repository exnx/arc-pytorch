import os
import urllib.request
import numpy as np
import zipfile
from PIL import Image

from scipy.ndimage import imread

data_dir = ("font_imgs")

def images_folder_to_NDarray(path_im):
    alphbts = os.listdir(path_im)
    ALL_IMGS = []

    # loop thru all alphbt directories
    for alphbt in alphbts:

        if os.path.isdir(os.path.join(path_im, alphbt)): # check if a alphbt is dir, otherwise ignore

            print('is a dir:', alphbt)

            chars = os.listdir(os.path.join(path_im, alphbt))  # returns all files in the char directory

            for char in chars:

                if os.path.isdir(os.path.join(path_im, alphbt, char)): # need to make sure char is a dir

                    img_filenames = os.listdir(os.path.join(path_im, alphbt, char))   # bug here....

                    char_imgs = []
                    for img_fn in img_filenames:

                        # need to check if img_fn is a .png
                        if img_fn.endswith('.png'):

                            fn = os.path.join(path_im, alphbt, char, img_fn)

                            # Open and convert to grayscale here
                            I = Image.open(fn).convert('L')  # ensure a gray image
                            I = np.asarray(I)  # convert to numpy
                            # I = imread(fn)  # alternate between this and PIL image open

                            I = np.invert(I)
                            char_imgs.append(I)

                        else:
                            print('is not a dir:', img_fn)

                    ALL_IMGS.append(char_imgs)

                else:
                    print('is not a dir:', char)

        else:
            print('not a dir:', alphbt)

    # print('ALL_IMGS shape', np.array(ALL_IMGS).shape)

    return np.array(ALL_IMGS)


def save_to_numpy() -> None:
    print("Converting folder {} to numpy array...".format(data_dir))
    np_array = images_folder_to_NDarray(data_dir)

    print('shape of all images', np_array.shape)

    np.save(os.path.join(data_dir, "font_imgs.npy"), np_array)
    print("Done.")


def main():
    save_to_numpy()

if __name__ == "__main__":
    main()
