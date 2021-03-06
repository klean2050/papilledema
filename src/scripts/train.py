from __future__ import print_function
from __future__ import division
import torch, torch.nn as nn, time
import os, copy, numpy as np
import matplotlib.pyplot as plt, cv2 as cv
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.model import *

cosine = lambda x, y: 1.0 - F.cosine_similarity(x, y)
contrast1 = nn.TripletMarginWithDistanceLoss(distance_function=cosine)
contrast2 = nn.TripletMarginWithDistanceLoss(distance_function=cosine)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, lens, num_epochs, mode="multiple"):

    since = time.time()
    val_acc_history, patience, best_loss = [], 5, 5
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):

        vprint = print if not epoch % 1 else lambda *a, **k: None
        vprint("\nEpoch {}/{}".format(epoch, num_epochs))
        vprint("-" * 11)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels, _ in dataloaders[phase]:
                for img in range(len(inputs)):
                    inputs[img] = inputs[img].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    if mode == "multiple":
                        out1, out2, out3, outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss2 = contrast1(out1, out2, out2[torch.randperm(out2.size()[0])])
                        loss3 = contrast2(out1, out3, out3[torch.randperm(out3.size()[0])])
                        if epoch < 5:
                            loss = 0.5 * loss2 + 0.5 * loss3
                        else:
                            loss = 0.9 * loss + 0.05 * loss2 + 0.05 * loss3
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / lens[phase]
            epoch_acc = running_corrects.double() / lens[phase]
            vprint("{} loss: {:.4f} Acc: {:.3f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "valid":
                val_acc_history.append(epoch_acc)
                if epoch_loss < best_loss:
                    if epoch < 5 and mode == "multiple":
                        continue
                    patience = 5
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience -= 1

        if not patience:
            break

    time_elapsed = time.time() - since
    print("\nTraining complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # save and load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataset, mode="multiple", sal=False):
    def get_saliency(im, target, name, code):
        saliency, _ = torch.max(im.grad.data.abs(), dim=1)
        saliency = saliency.reshape(224, 224).cpu().detach().numpy()
        t = 2 * np.mean(saliency.ravel())
        _, img = cv.threshold(saliency, t, 1, cv.THRESH_TOZERO)
        plt.imsave(f"data/saliency/sal{target}_{name}_{code}.png", img, cmap="hot")
        return saliency

    set_parameter_requires_grad(model, True)
    model.eval()

    outs, targets = [], []
    for inputs, target, name in dataset:
        for img in range(len(inputs)):
            inputs[img] = inputs[img].to(device)
            inputs[img] = inputs[img].unsqueeze(dim=0)
            inputs[img].requires_grad = True

        if mode == "multiple":
            _, _, _, out = model(inputs)
        else:
            out = model(inputs)

        output_idx = out.argmax()
        output_max = out[0, output_idx]
        output_max.backward(retain_graph=True)

        _, out = torch.max(out, 1)
        outs.append(out.item())
        targets.append(target)

        if sal:
            sal0 = get_saliency(inputs[0], target, name, 0)
            sal1 = get_saliency(inputs[1], target, name, 1)
            sal2 = get_saliency(inputs[2], target, name, 2)

    auc = roc_auc_score(targets, outs)
    ci = 1.96 * np.sqrt((auc * (1 - auc)) / len(outs))
    print("\n" + classification_report(targets, outs, digits=3))
    print("AUC: {:.3f} +/- {:.3f}".format(auc, np.clip(ci, 0, 1)))


def train_fold(root, subjects, batch_size, epochs):

    train_s, valid_s, y, _ = train_test_split(
        [v[0] for v in list(set(subjects))], [v[1] for v in list(set(subjects))], test_size=0.2
    )

    weights = [len(y) / (len(y) - sum(y)), len(y) / sum(y)]
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    tr_dataset = PapDataset(root, train_s, train=True)
    te_dataset = PapDataset(root, valid_s, train=False)

    trainloader = DataLoader(tr_dataset, batch_size=batch_size, num_workers=4)
    testloader = DataLoader(te_dataset, batch_size=batch_size, num_workers=4)
    dataloaders_dict = {"train": trainloader, "valid": testloader}
    len_dict = {"train": len(tr_dataset), "valid": len(te_dataset)}

    model_ft = MultiBranchCNN(False, use_pretrained=True, branches=3)
    model_ft = model_ft.to(device)
    optimizer = optim.AdamW(model_ft.parameters(), lr=1e-4)

    model_ft_m, _ = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer,
        lens=len_dict,
        num_epochs=epochs,
        mode="multiple",
    )
    test_model(model_ft_m, te_dataset, mode="multiple")

