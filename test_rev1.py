import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
from torchvision import transforms

from models import ArcBinaryClassifier
from batcher import Batcher


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')
parser.add_argument('--img1', required=True, help='path to first image to compare')
parser.add_argument('--img2', required=True, help='path to second image to compare.')
# parser.add_argument('--same', action='store_true', help='whether to generate same character pairs or not')

opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

# make directory for storing images.
images_path = os.path.join("visualization", opt.name)
os.makedirs(images_path, exist_ok=True)


# initialise the batcher
batcher = Batcher(batch_size=opt.batchSize)


def display(image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

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

    plt.savefig(os.path.join(images_path, name))


def get_sample(discriminator):

    # batching....

    mean_pixel = 0.08051840363667802

    SIZE = 32

    path_img1 = opt.img1
    path_img2 = opt.img2

    img1_img = Image.open(path_img1)
    img2_img = Image.open(path_img2)
    #
    # img1_img = np.asarray(img1_img.resize((SIZE, SIZE)), dtype=np.float32)
    # img2_img = np.asarray(img1_img.resize((SIZE, SIZE)), dtype=np.float32)
    #
    # print('img', img1_img)
    #
    #
    # print('image shape', img1_img.shape)
    # print('image pixel value', img1_img)


    # print('image size', img1_img.size)

    transform = transforms.Compose([
        transforms.Resize(size=SIZE),  # resize first
        transforms.ToTensor()  # convert to tensor
    ])

    img1_tensor = transform(img1_img)
    img2_tensor = transform(img2_img)

    print(img1_tensor.shape)

    # img1_tensor = torch.from_numpy(img1_img)
    # img2_tensor = torch.from_numpy(img2_img)

    X = torch.stack([img1_tensor, img2_tensor], dim=1)  # (B, 2, h, w)  # array of image PAIRS now!!

    X = X - mean_pixel

    # end batching...

    pred = discriminator(X)

    print('X shape',  X.shape)

    print('pred', pred)
    print('pred shape', pred.shape)

    # print('Y true score', Y[index])

    # print('X shape', X.shape)

    # should return 2 x Size x Size (select one from Batch, i.e. the first)
    return X[0]

    # # size of the set to choose sample from from
    # sample_size = 30
    # X, Y = batcher.fetch_batch("train", batch_size=sample_size)
    # pred = discriminator(X)
    #
    # if opt.same:
    #     same_pred = pred[sample_size // 2:].data.numpy()[:, 0]
    #     mx = same_pred.argsort()[len(same_pred) // 2]  # choose the sample with median confidence
    #     index = mx + sample_size // 2
    # else:
    #     diff_pred = pred[:sample_size // 2].data.numpy()[:, 0]
    #     mx = diff_pred.argsort()[len(diff_pred) // 2]  # choose the sample with median confidence
    #     index = mx
    #
    # return X[index]


def visualize():

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))

    arc = discriminator.arc

    sample = get_sample(discriminator)

    # print('sample shape', sample.shape)

    all_hidden = arc._forward(sample[None, :, :])[:, 0, :]  # (2*numGlimpses, controller_out)
    glimpse_params = torch.tanh(arc.glimpser(all_hidden))
    masks = arc.glimpse_window.get_attention_mask(glimpse_params, mask_h=opt.imageSize, mask_w=opt.imageSize)

    # separate the masks of each image.
    masks1 = []
    masks2 = []

    for i, mask in enumerate(masks):
        if i % 2 == 1:  # the first image outputs the hidden state for the next image
            masks1.append(mask)
        else:
            masks2.append(mask)

    for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
        display(sample[0], mask1, sample[1], mask2, "img_{}".format(i))
        print('made it here too!')

if __name__ == "__main__":
    visualize()
