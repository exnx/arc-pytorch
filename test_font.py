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
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')


parser.add_argument('--load', required=True, help='the model to load from.')



# parser.add_argument('--img1', required=True, help='path to first image to compare')
# parser.add_argument('--img2', required=True, help='path to second image to compare.')
# parser.add_argument('--img3', required=True, help='path to third image to compare')
# parser.add_argument('--img4', required=True, help='path to fourth image to compare.')
# parser.add_argument('--same', action='store_true', help='whether to generate same character pairs or not')


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

# # make directory for storing images.
# images_path = os.path.join("test_visual", opt.name)
# os.makedirs(images_path, exist_ok=True)


# initialise the batcher
batcher = Batcher(batch_size=opt.batchSize)


def display(image1, mask1, image2, mask2, sample_pred, sample_target, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    label = 'Target: {}, Pred: {}'.format(sample_target, sample_pred)

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

# def read_img(img_path):
#     img = Image.open(img_path).convert('L')  # ensure a gray image
#     img_np = np.asarray(img)  # convert to numpy
#     img_np = np.invert(img)  # flip black and white
#     img_rs = resize(img_np, (opt.imageSize, opt.imageSize)).astype("float32")

#     return img_rs

# def _fetch_batch():
#     mean_pixel = 0.08051840363667802
#     image_size = opt.imageSize
#     batch_size = opt.batchSize

#     X = np.zeros((2 * batch_size, image_size, image_size), dtype='uint8')

#     for i in range(batch_size // 2):
#         # choose similar chars.  choose char idx from start to end of alphabet idxs
#         same_idx = choice(range(starts[0], starts[-1] + sizes[-1]))

#         # choose dissimilar chars within alphabet
#         alphbt_idx = choice(num_alphbts, p=p)
#         char_offset = choice(sizes[alphbt_idx], 2, replace=False)  # np array of 2 numbers
#         diff_idx = starts[alphbt_idx] + char_offset  # starts = all alphabet start idxs, so 2 offsets gives 2 diff_idxs

#         X[i], X[i + batch_size] = data[diff_idx, choice(self.num_samples_per_char, 2)]  # 2 diff idx and 2 nums between 0-19 gives 2 diff chars imgs
#         X[i + batch_size // 2], X[i + 3 * batch_size // 2] = data[same_idx, choice(self.num_samples_per_char, 2, replace=False)]  # chooses same char within alphabet

#     y = np.zeros((batch_size, 1), dtype='int32')
#     y[:batch_size // 2] = 0  # first half are diff imgs
#     y[batch_size // 2:] = 1  # second half are same imgs

#     X = X / 255.0

#     X = X - mean_pixel

#     # print('mean pixel', self.mean_pixel)
#     # print('batch size', batch_size)

#     # print('X before new axis in _fetch batch', X.shape)

#     X = X[:, np.newaxis]
#     X = X.astype("float32")

#     # print('X shape in _fetch batch', X.shape)

#     return X, y

# def fetch_batch():
#     image_size = opt.imageSize
#     batch_size = opt.batchSize

#     X = _fetch_batch()

#     X = torch.from_numpy(X).view(2*batch_size, image_size, image_size)

#     # should be 4, 32, 32
#     X1 = X[:batch_size]  # (B, h, w)
#     X2 = X[batch_size:]  # (B, h, w)

#     X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)  # array of image PAIRS now!!

#     return X


# def get_sample(discriminator, batch_size=2):

#     X = fetch_batch()
#     pred = discriminator(X)
#     # should return 2 x Size x Size (select one from Batch, i.e. the first)
#     return X[0]

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
