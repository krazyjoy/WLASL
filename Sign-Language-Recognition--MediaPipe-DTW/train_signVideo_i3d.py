import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
import torch
import random
# from datasets.nslt_dataset import NSLT as Dataset
from SignVideoDataset import SignVideoDataset as Dataset

import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def write_results(train_file="training_results.txt", val_file="validation_results.txt", 
    mode="train", epoch=None, steps=None, 
    train_acc=None, train_loc_loss=None, train_cls_loss=None, train_tot_loss=None,
    val_acc=None, val_loc_loss=None, val_cls_loss=None, val_tot_loss=None):
    if mode=="train":
        with open(train_file, "a") as f:
            f.write(f"{epoch}, {steps}, {train_acc}, {train_loc_loss}, {train_cls_loss}, {train_tot_loss}\n")
    elif mode=="val":
        with open(val_file, "a") as f:
            f.write(f"{epoch}, {steps}, {val_acc}, {val_loc_loss}, {val_cls_loss}, {val_tot_loss}\n")

def plot_loss_n_acc(epoch: list, steps: list, mode:str, train_acc_history: list=None, train_tot_loss_history:list=None, val_acc_history: list=None, val_tot_loss_history: list=None):
    if mode=="train":
        plt.figure(figsize=(21, 6))
        plt.plot(steps,  train_acc_history, label="train accuracy", marker="o", markersize=8)
        plt.xlabel("steps")
        plt.title("training accuracy per steps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_accuracy_plot.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(21, 6))
        plt.plot(steps,  train_tot_loss_history, label="train loss", marker="o", markersize=8)
        plt.xlabel("steps")
        plt.title("training loss per steps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_loss_plot.png")
        plt.show()
        plt.close()

    elif mode=="val":
        plt.figure(figsize=(21, 6))
        plt.plot(epoch,  val_acc_history, label="val accu", marker="x", markersize=8)
        plt.xlabel("epoch")
        plt.title("val accuracy per epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("valiation_accuracy_plot.png")
        plt.show()
        plt.close()
        
        plt.figure(figsize=(21, 6))
        plt.plot(epoch,  val_tot_loss_history, label="val loss", marker="x", markersize=8)
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("valiation_loss_plot.png")
        plt.show()
        plt.close()

def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):


    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])


    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    print("len(dataset): ", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)
    
    
    val_dataset = Dataset(train_split, 'val', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('../code/I3D/weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('../code/I3D/weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    print("num_classes: ", num_classes)
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    print("num_steps_per_update: ", num_steps_per_update)
    best_val_score = 0

    # plot
    train_acc_history = []
    val_acc_history = []
    train_loc_loss_history = []
    val_loc_loss_history = []
    train_cls_loss_history = []
    val_cls_loss_history = []
    train_tot_loss_history = []
    val_tot_loss_history = []
    train_steps_history = []
    train_epoch_history = []
    val_steps_history = []
    val_epoch_history = []

    training_result = "training_results.txt"
    with open(training_result, "w") as train_f:
        train_f.write("epoch, steps, train_acc, train_loc_loss, train_cls_loss, train_tot_loss\n")
        train_f.close()

    val_result = "validation_results.txt"
    with open(val_result, "w") as val_f:
        val_f.write("epoch, steps, val_acc, val_loc_loss, val_cls_loss, val_tot_loss\n")
        val_f.close()


    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            collected_vids = []

            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int16)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 50 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        loc_loss = tot_loc_loss / (50 * num_steps_per_update)
                        cls_loss = tot_cls_loss / (50 * num_steps_per_update)
                        total_loss = tot_loss / 50
                        print(
                            'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 phase,
                                                                                                                 loc_loss,
                                                                                                                 cls_loss,
                                                                                                                 total_loss,
                                                                                                                 acc))
                        train_epoch_history.append(epoch)
                        train_steps_history.append(steps)
                        train_acc_history.append(acc)
                        train_loc_loss_history.append(loc_loss)
                        train_cls_loss_history.append(cls_loss)
                        train_tot_loss_history.append(total_loss)
                        write_results(mode='train', epoch=epoch, steps= steps, train_acc=acc, train_loc_loss=loc_loss, train_cls_loss=cls_loss, train_tot_loss=total_loss)
                        plot_loss_n_acc(mode="train", epoch=train_epoch_history, steps=train_steps_history, train_acc_history=train_acc_history, train_tot_loss_history=train_tot_loss_history)
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                        

            if phase == 'val':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                                   6) + '_%3f.pt' % val_score

                    torch.save(i3d.module.state_dict(), model_name)
                    print(model_name)

                val_loc_loss = tot_loc_loss / num_iter
                val_cls_loss = tot_cls_loss / num_iter
                val_tot_loss = (tot_loss * num_steps_per_update) / num_iter
                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))
                val_epoch_history.append(epoch)
                val_steps_history.append(steps)
                val_acc_history.append(val_score)
                val_loc_loss_history.append(val_loc_loss)
                val_cls_loss_history.append(val_cls_loss)
                val_tot_loss_history.append(val_tot_loss)
                write_results(mode='val', epoch=epoch, steps=steps, val_acc=val_score, val_loc_loss=val_loc_loss, val_cls_loss=val_cls_loss, val_tot_loss=val_tot_loss)
                plot_loss_n_acc(mode="val", epoch=val_epoch_history, steps=val_steps_history, val_acc_history=val_acc_history, val_tot_loss_history=val_tot_loss_history)
                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    # WLASL setting
    mode = 'rgb'
    root = {'word': './data/videos'}
    save_model = 'checkpoints/'
    train_split = "./mediapipe_dataset.json"

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    weights = None
    # weights = 'checkpoints/nslt_2000_002882_0.261364.pt'
    config_file = '../code/I3D/configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
