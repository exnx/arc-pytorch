import os
import argparse
import torch
from torch.autograd import Variable
from datetime import datetime, timedelta

import font_batcher
from font_batcher import Batcher
import models
from models import ArcBinaryClassifier

import multiprocessing
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=8, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')


def get_pct_accuracy(pred: Variable, target) -> int:
    hard_pred = (pred > 0.5).int()
    # correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy


def train():
    opt = parser.parse_args()

    if opt.cuda:
        batcher.use_cuda = True
        models.use_cuda = True

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)
    os.makedirs(models_path, exist_ok=True)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)

    if opt.cuda:
        discriminator.cuda()

    # load from a previous checkpoint, if specified.
    if opt.load is not None:
        discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr)

    # load the dataset in memory.
    loader = Batcher(batch_size=opt.batchSize, image_size=opt.imageSize)

    # ready to train ...
    best_validation_loss = None
    saving_threshold = 1.02
    last_saved = datetime.utcnow()
    save_every = timedelta(minutes=10)

    i = -1


    while True:
        i += 1


        X, Y = loader.fetch_batch("train")
        discriminator.train()  # set to train mode
        pred = discriminator(X)
        loss = bce(pred, Y.float())

        # note that this only validating every 10 steps of training, need to fix this later 
        # need to set in eval mode and turn off gradients in eval mode
        
        with torch.no_grad():
            if i % 10 == 0:

                # validate your model
                X_val, Y_val = loader.fetch_batch("val")
                discriminator.eval()  # set to evaluation mode
                pred_val = discriminator(X_val)
                loss_val = bce(pred_val, Y_val.float())

                training_loss = loss.item()
                validation_loss = loss_val.item()

                print("Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}".format(
                    i, get_pct_accuracy(pred, Y), training_loss, get_pct_accuracy(pred_val, Y_val), validation_loss
                ))

                if best_validation_loss is None:
                    best_validation_loss = validation_loss

                if best_validation_loss > (saving_threshold * validation_loss):
                    print("Significantly improved validation loss from {} --> {}. Saving...".format(
                        best_validation_loss, validation_loss
                    ))
                    discriminator.save_to_file(os.path.join(models_path, str(validation_loss)))
                    best_validation_loss = validation_loss
                    last_saved = datetime.utcnow()

                if last_saved + save_every < datetime.utcnow():
                    print("It's been too long since we last saved the model. Saving...")
                    discriminator.save_to_file(os.path.join(models_path, str(validation_loss)))
                    last_saved = datetime.utcnow()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main() -> None:

    p = multiprocessing.Process(target=train, name="Train")
    p.start()

    time.sleep(14400)  # 14400 secs for 4 hrs

    # If thread is active
    if p.is_alive():
        print("train is running... let's kill it...")

        # Terminate foo
        p.terminate()


if __name__ == "__main__":
    main()
