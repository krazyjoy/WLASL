import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation
import pandas as pd
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def run(split_file, pose_data_root, configs, subset, save_model_to=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='k_copies')
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=6, pin_memory=False)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # setup the model input: (batch, nodes, num_samples * 2)
    model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    # checkpoint = torch.load("/home/chuan194/work/roboticVision/WLASL/code/TGCN/checkpoints/asl2000/gcn_epoch=16_train_acc=4.815187546537603.pth", map_location=device)
    # model.load_state_dict(checkpoint)
    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    # optimizer = optim.SGD(vgg_gru.parameters(), lr=lr, momentum=0.00001)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    df = pd.DataFrame(columns=["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

    # record training process
    epochs_range = df['Epoch'].to_list()
    epoch_train_losses = df['Train Loss'].to_list()
    epoch_train_scores = df['Train Accuracy'].to_list()
    epoch_val_losses = df['Val Loss'].to_list()
    epoch_val_scores = df['Val Accuracy'].to_list()

    best_test_acc = 0
    # start training
    for epoch in range(1, int(epochs)):
        # train, test model

        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)

        # torch.save(model.state_dict(), os.path.join('checkpoints', subset, 'gcn_epoch={}_train_acc={}.pth'.format(epoch, 100*np.mean(train_scores))))
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)

        # scheduler.step(val_loss)
        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(np.mean(train_losses))
        epoch_train_scores.append(100*np.mean(train_scores))
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(100*val_score[0])


        # epochs_range = list(range(1, epoch+2))
        epochs_range.append(epoch)
        df = pd.DataFrame({
            "Epoch": epochs_range,
            "Train Loss": epoch_train_losses,
            "Train Accuracy": epoch_train_scores,
            "Val Loss": epoch_val_losses,
            "Val Accuracy": epoch_val_scores
        })
        df = df.reset_index(drop=True)
        df.to_csv(f"./output/tgcn_training_log_{subset}.csv", index=False)

        plt.suptitle('TGCN Training & Validation Trend', fontsize=22) 
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, epoch_train_losses, label='Train Loss')
        plt.plot(epochs_range, epoch_val_losses, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss over Epochs', fontsize=18)
        plt.legend(fontsize=14)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, epoch_train_scores, label='Train Accuracy')
        plt.plot(epochs_range, epoch_val_scores, label='Validation Accuracy')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Accuracy over Epochs', fontsize=16)
        plt.legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
        plt.savefig(f"./output/tgcn_training_log_{subset}.png")
        plt.close()


        if val_score[0] > best_test_acc:
            best_test_acc = 100*val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(), os.path.join('checkpoints', subset, 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))

    # utils.plot_curves()

    # class_names = train_dataset.label_encoder.classes_
    # utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
    #                             save_to=f'output/train-conf-mat-{subset}')
    # utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to=f'output/val-conf-mat-{subset}')


if __name__ == "__main__":
    root = '/home/chuan194/work/roboticVision/WLASL'
    subset = 'asl2000'

    split_file = os.path.join(root, 'data/splits/{}.json'.format(subset))
    
    pose_data_root = os.path.join(root, 'data/pose_per_individual_videos')

    config_file = os.path.join(root, 'code/TGCN/configs/{}.ini'.format(subset))
    configs = Config(config_file)

    logging.basicConfig(filename='output/{}.log'.format(subset, os.path.basename(config_file)[:-4]), level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(split_file=split_file, configs=configs, pose_data_root=pose_data_root, subset=subset)
    logging.info('Finished main.run()')
    # utils.plot_curves()
