from __future__ import print_function
from __future__ import division
import torch, torch.nn as nn
import os, copy, numpy as np
import matplotlib.pyplot as plt, cv2 as cv
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from src.model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, lens, num_epochs):

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

            running_loss, preds, targets = 0.0, [], []
            for inputs, labels, names in dataloaders[phase]:
                labels, names = labels.to(device), names.to(device)
                for img in range(len(inputs)):
                    inputs[img] = inputs[img].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, dd_out = model(inputs)
                    _, pred = torch.max(outputs, 1)

                    ce_loss = criterion(outputs, labels)
                    dd_crt = nn.CrossEntropyLoss()
                    dd_loss = dd_crt(dd_out, names)
                    loss = 0.8 * ce_loss + 0 * dd_loss

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs[0].size(0)
                    preds.extend(pred.cpu().detach().numpy())
                    targets.extend(labels.cpu().detach().numpy())

            epoch_loss = running_loss / lens[phase]
            epoch_acc = accuracy_score(targets, preds)
            vprint("{} loss: {:.4f} Acc: {:.3f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "valid":
                val_acc_history.append(epoch_acc)
                if epoch_loss < best_loss:
                    patience = 5
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience -= 1

        if not patience:
            break

    # save and load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataset, fold):

    model.eval()
    outs, targets = [], []
    for inputs, target, _ in dataset:
        for img in range(len(inputs)):
            inputs[img] = inputs[img].to(device)
            inputs[img] = inputs[img].unsqueeze(dim=0)

        out, _ = model(inputs)
        _, out = torch.max(out, 1)
        outs.append(out.item())
        targets.append(target)

    def calc_ci(num):
        ci = np.sqrt((num * (1 - num)) / len(outs))
        ci = np.clip(1.96 * ci, 0, 1)
        return "95% CI, {:.1f} to {:.1f}".format(100.0 * (num - ci), 100 * (num + ci))

    auc = roc_auc_score(targets, outs)
    report = classification_report(targets, outs, output_dict=True)
    sns, spc = report["0"]["recall"], report["1"]["recall"]

    os.makedirs("results", exist_ok=True)
    with open("results/this_logger.txt", "a") as f:
        f.write("\n" + str(fold))
        f.write("\nAUC: {:.1f}% ({})".format(100.0 * auc, calc_ci(auc)))
        f.write("\nSensitivity: {:.1f}% ({})".format(100.0 * sns, calc_ci(sns)))
        f.write("\nSpecificity: {:.1f}% ({})".format(100.0 * spc, calc_ci(spc)))


def get_saliency(model, dataset):
    def sal(im, target, name, code):
        saliency, _ = torch.max(im.grad.data.abs(), dim=1)
        saliency = saliency.reshape(224, 224).cpu().detach().numpy()
        t = 2 * np.mean(saliency.ravel())
        _, img = cv.threshold(saliency, t, 1, cv.THRESH_TOZERO)

        os.makedirs("results/saliency", exist_ok=True)
        plt.imsave(f"results/saliency/sal{target}_{name}_{code}.png", img, cmap="hot")

    set_parameter_requires_grad(model, True)
    model.eval()

    for inputs, target, name in dataset:
        for img in range(len(inputs)):
            inputs[img] = inputs[img].to(device)
            inputs[img] = inputs[img].unsqueeze(dim=0)
            inputs[img].requires_grad = True

        out, _ = model(inputs)
        output_idx = out.argmax()
        output_max = out[0, output_idx]
        output_max.backward(retain_graph=True)

        sal(inputs[0], target, name, 0)
        sal(inputs[1], target, name, 1)
        sal(inputs[2], target, name, 2)


def train_fold(root, subjects, batch_size, epochs, mode, fold):

    train_s, valid_s, y, _ = train_test_split(
        [v[0] for v in list(set(subjects))],
        [v[1] for v in list(set(subjects))],
        test_size=0.15,
        random_state=(fold + 0) * 43,
    )
    """
    train_s = [s[0] for s in subjects if s[2] != 6.0]
    valid_s = [s[0] for s in subjects if s[2] == 6.0]
    y = [s[1] for s in subjects if s[2] != 6.0]
    """

    weights = [len(y) / (len(y) - sum(y)), len(y) / sum(y)]
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    tr_dataset = PapDataset(root, train_s, train=True)
    te_dataset = PapDataset(root, valid_s, train=False)
    trainloader = DataLoader(tr_dataset, batch_size=batch_size, num_workers=4)
    testloader = DataLoader(te_dataset, batch_size=batch_size, num_workers=4)

    dataloaders_dict = {"train": trainloader, "valid": testloader}
    len_dict = {"train": len(tr_dataset), "valid": len(te_dataset)}

    if mode == "single":
        model_ft = SingleBranchCNN(False, use_pretrained=True, subs=len(subjects))
    else:
        model_ft = MultiBranchCNN(False, use_pretrained=True, subs=len(subjects))
    model_ft = model_ft.to(device)

    optimizer = optim.AdamW(model_ft.parameters(), lr=2e-5)
    model_ft_m, _ = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer,
        lens=len_dict,
        num_epochs=epochs,
    )
    test_model(model_ft_m, te_dataset, fold)
