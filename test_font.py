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


opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

def get_pct_accuracy(pred, target):
    hard_pred = (pred > 0.5).int()
    # correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def test(epochs=1):

    # set up the optimizer.
    bce = torch.nn.BCELoss()


    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)

    if opt.cuda:
        bce = bce.cuda()
        discriminator.cuda()
    
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))

    arc = discriminator.arc

    # load the dataset in memory.
    loader = Batcher(batch_size=opt.batchSize, image_size=opt.imageSize)

    # retrieve total number of samples
    num_unique_chars = loader.data.shape[0]
    num_samples_per_char = loader.data.shape[1]
    total_num_samples = num_unique_chars * num_samples_per_char

    # calc number of steps per epoch
    num_batches_per_epoch = int(total_num_samples / opt.batchSize)

    print('num_batches_per_epoch', num_batches_per_epoch)

    # loop thru epochs

    all_epoch_losses = []  # track all epoch losses here, for a whole batch
    all_epoch_acc = []

    # used for precision recall later
    preds = []
    labels = []

    discriminator.eval()  # set in eval mode

    # turn off gradients
    with torch.no_grad():

        for epoch in range(epochs):

            running_epoch_loss = 0
            running_epoch_acc = 0

            # loop through num of batches in an epoch
            for batch_num in range(num_batches_per_epoch):

                X, Y = loader.fetch_batch("test")  # loader loads data to cuda if available

                pred = discriminator(X)
                batch_loss = bce(pred, Y.float())
                running_epoch_loss += batch_loss.item()  # sum all the loss

                batch_acc = get_pct_accuracy(pred, Y)
                running_epoch_acc += batch_acc

                if batch_num % 100 == 0:
                    print("Batch: {} \t Test: Acc={}%, Loss={}:".format(batch_num, batch_acc, batch_loss))

            # append the average loss over the epoch (for a whole batch)
            all_epoch_losses.append(running_epoch_loss / num_batches_per_epoch)
            all_epoch_acc.append(running_epoch_acc / num_batches_per_epoch)

        # loop through losses and display per epoch
        for i in range(len(all_epoch_losses)):
            print('Epoch {} - Loss: {}, Accuracy: {}'.format(i, all_epoch_losses[i], all_epoch_acc[i]))




def main():
    test()

if __name__ == "__main__":
    main()