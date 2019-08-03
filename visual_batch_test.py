import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
from torchvision import transforms
from numpy.random import choice
from models import ArcBinaryClassifier
import font_batcher
from font_batcher import Batcher
from scipy.ndimage import imread
import scipy.misc
from scipy.misc import imresize as resize


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

# most important args for running visual test
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')

# make font_imgs root
images_path = './test_visuals'
if not os.path.exists(images_path):
    os.mkdir(images_path)

opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

# initialise the batcher
batcher = Batcher(batch_size=opt.batchSize)


def display(image1, mask1, image2, mask2, sample_pred, sample_target, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    label = 'Target: {}, Pred: {:.3f}'.format(sample_target, sample_pred)

    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.numpy(), cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.numpy(), cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.title(label, fontdict=None, loc='center', pad=None)

    plt.savefig(os.path.join(images_path, name))

def visualize():

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))

    arc = discriminator.arc

    # load the dataset in memory.
    loader = Batcher(batch_size=opt.batchSize, image_size=opt.imageSize)

    X, Y = loader.fetch_batch("test")
    pred = discriminator(X)
    loss = bce(pred, Y.float())

    for sample_num, sample in enumerate(X):

        all_hidden = arc._forward(sample[None, :, :])[:, 0, :]  # (2*numGlimpses, controller_out)
        glimpse_params = torch.tanh(arc.glimpser(all_hidden))
        masks = arc.glimpse_window.get_attention_mask(glimpse_params, mask_h=opt.imageSize, mask_w=opt.imageSize)

        sample_pred = pred[sample_num].item()
        sample_target = Y[sample_num].item()

        # separate the masks of each image.
        masks1 = []
        masks2 = []

        for i, mask in enumerate(masks):
            if i % 2 == 1:  # the first image outputs the hidden state for the next image
                masks1.append(mask)
            else:
                masks2.append(mask)

        for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
            display(sample[0], mask1, sample[1], mask2, sample_pred, sample_target, "sample_{}_img_{}".format(sample_num, i))
            # print('made it here too!')

if __name__ == "__main__":
    visualize()
