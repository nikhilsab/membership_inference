import numpy as np
import sys
from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Subset
import torchvision.models as models
from model import *
from sklearn.metrics import roc_auc_score, roc_curve, auc


def train(train_model, epochs, opt, criterion, scheduler, train_loader, val_loader, save_name):
    for epoch in range(0, epochs):
        train_model.train()
        loss_list = []
        train_acc_list = []
        val_acc_list = []
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            output_train = train_model(images)
#             print(output_train, output_train.shape, labels, labels.shape)
            _, pred_train = torch.max(output_train.data, 1)
            correct_train = (pred_train == labels).sum().item()
            train_acc = correct_train / labels.size(0)
            train_acc_list.append(float(train_acc))
            loss = criterion(output_train, labels)
            loss.backward()
            loss_list.append(float(loss.data))
            opt.step()

        train_model.eval()
        for images,labels in val_loader:
            with torch.no_grad():
                images, labels = images.cuda(), labels.cuda()
                output_val = train_model(images)
                _, pred_val = torch.max(output_val.data, 1)
                correct_val = (pred_val == labels).sum().item()
                val_acc = correct_val / labels.size(0)
                val_acc_list.append(float(val_acc))

        scheduler.step()

        ave_loss = np.average(np.array(loss_list))
        ave_train_acc = np.average(np.array(train_acc_list))
        ave_val_acc = np.average(np.array(val_acc_list))                            
        print('Epoch:%d, Loss: %.03f Train accuracy: %.02f Test accuracy: %.02f' % (epoch+1, ave_loss,100.*ave_train_acc,100.*ave_val_acc))
    #Save the surrogate model
    save_path = './checkpoint/' + (save_name) + '_' + str(epochs) + '.pth'
    torch.save(train_model.state_dict(),save_path)
    
    
def eval_(eval_model, test_loader,criterion):

    eval_model.eval()
    predictions = []
    loss = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = eval_model(data)
            loss_fn = criterion(output, target)
            prob = F.softmax(output, dim=1)
            targets.append(target.detach().cpu())
            predictions.append(prob.detach().cpu().numpy())
            loss.append(-loss_fn.detach().cpu()) 
    
    return predictions, targets, loss


def _Mentr(preds, y):
    fy = np.sum(preds*y, axis=1)
    fi = preds*(1-y)
    score = -(1-fy)*np.log(fy+1e-30)-np.sum(fi*np.log(1-fi+1e-30), axis=1)
    return score


def mia(groundtruth, predictions, targets, loss, test_loader, score):

    fpr, tpr, _ = roc_curve(groundtruth, score)
    mia = auc(fpr, tpr)
    low = tpr[np.where(fpr<.001)[0][-1]]
    acc = np.max(1-(fpr+(1-tpr))/2)
    return mia, low, acc, fpr, tpr

def metric_scores(predictions, targets, loss, test_loader):
    
    pred=np.reshape(predictions, (len(test_loader.dataset),10))
    max_score=np.max(pred, axis=1)
    loss_score=np.array(loss)
    
    tar = np.array(targets)
    target_categorical = np.zeros((tar.size, tar.max() + 1))
    target_categorical[np.arange(tar.size), tar] = 1
    
    loss_score=np.array(loss)
    max_score=np.max(pred, axis=1)
    _Mentr_score = -_Mentr(pred,target_categorical)
    
    return loss_score, max_score, _Mentr_score