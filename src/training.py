from __future__ import print_function
from __future__ import division
import torch, torch.nn as nn
import os, copy, numpy as np
import matplotlib.pyplot as plt, cv2 as cv
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from .utils import *
from .models import *
from .dataset import PapDataset

os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, lens):

    optimizer = optim.AdamW(model.parameters(), lr=learning_r)
    val_acc_history, patience, best_loss = [], 5, 5
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):

        vprint = print if not epoch % 1 else lambda *a, **k: None
        vprint("\nEpoch {}/{}".format(epoch, num_epochs))
        vprint("-" * (9 + len(str(epoch))))

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, preds, targets = 0.0, [], []
            for inputs, labels, names, sites in dataloaders[phase]:
                # print(names)
                labels = labels.to(device)
                # names = names.to(device)
                sites = sites.to(device)
                for img in range(len(inputs)):
                    inputs[img] = inputs[img].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, dd_out = model(inputs)
                    _, pred = torch.max(outputs, 1)

                    ce_loss = criterion(outputs, labels)
                    dd_crt = nn.CrossEntropyLoss()
                    if ddloss == "sites" and phase == "train":
                        dd_loss = dd_crt(dd_out, sites)
                    elif ddloss == "names":
                        dd_loss = dd_crt(dd_out, names)
                    else:
                        dd_loss = 0
                    loss = ce_loss + 0.01 * dd_loss

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


def test_model(model, loader, raw):

    model.eval()
    m = nn.Softmax(dim=0)
    outs, targets = [], []
    for inputs, labels, names, _ in loader:
        for img in range(len(inputs)):
            inputs[img] = inputs[img].to(device)

        with torch.no_grad():
            out, _ = model(inputs)

            # get raw scores for images
            for i, name in enumerate(names):
                pred = m(out[i]).tolist()[0]
                if name not in raw.keys():
                    raw[name] = [(pred, labels[i].item())]
                else:
                    raw[name].append((pred, labels[i].item()))

            _, out = torch.max(out, 1)
            outs.extend(out.cpu().detach().numpy())
            targets.extend(labels.cpu().detach().numpy())

    auc = roc_auc_score(targets, outs)
    acc = accuracy_score(targets, outs)
    report = classification_report(targets, outs, output_dict=True)
    sns, spc = report["0"]["recall"], report["1"]["recall"]
    return auc, acc, sns, spc, raw


def get_saliency(model, dataset):
    def sal(im, target, name, code, this_label):
        saliency, _ = torch.max(im.grad.data.abs(), dim=1)
        saliency = saliency.reshape(224, 224).cpu().detach().numpy()
        t = 2 * np.mean(saliency.ravel())
        _, img = cv.threshold(saliency, t, 1, cv.THRESH_TOZERO)

        os.makedirs("saliency", exist_ok=True)
        outname = f"sal{this_label}_{name}_TRUE={target}_{code}of3.png"
        if outname not in os.listdir("saliency"):
            plt.imsave("saliency/" + outname, img, cmap="hot")

    set_parameter_requires_grad(model)
    model.eval()

    for inputs, target, name, _ in dataset:
        for img in range(len(inputs)):
            inputs[img] = inputs[img].to(device)
            inputs[img] = inputs[img].unsqueeze(dim=0)
            inputs[img].requires_grad = True

        out, _ = model(inputs)

        # saliency map for PA
        output_max = out[0, 0]
        output_max.backward(retain_graph=True)

        sal(inputs[0], target, name, 1, 0)
        sal(inputs[1], target, name, 2, 0)
        sal(inputs[2], target, name, 3, 0)

        # saliency map for PPA
        output_max = out[0, 1]
        output_max.backward(retain_graph=True)

        sal(inputs[0], target, name, 1, 1)
        sal(inputs[1], target, name, 2, 1)
        sal(inputs[2], target, name, 3, 1)


def train_fold(subjects, fold, logger, raw):

    # repeat each fold for robustness
    aucs, accs, snss, spcs = [], [], [], []
    for _ in range(n_turns):

        if per_site:
            train_s = [s[0] for s in subjects if s[2] != test_site]
            valid_s = [s[0] for s in subjects if s[2] == test_site]
        elif external:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=29011997
            )
            train_i, valid_i = next(
                sss.split(
                    [v[0] for v in subjects],
                    [v[1] for v in subjects],
                    groups=[v[2] for v in subjects]
                )
            )
            train_s = [subjects[i][0] for i in train_i]
            valid_s = [subjects[i][0] for i in valid_i]
        else:
            train_s, valid_s, _, _ = train_test_split(
                [v[0] for v in subjects],
                [v[1] for v in subjects],
                test_size=test_size,
                random_state=fold * 44,
            )

        tr_dataset = PapDataset(data_dir, train_s, train=True)
        te_dataset = PapDataset(data_dir, valid_s, train=False)
        trainloader = DataLoader(tr_dataset, batch_size=batch_size, num_workers=4)
        testloader = DataLoader(te_dataset, batch_size=batch_size, num_workers=4)

        dataloaders_dict = {"train": trainloader, "valid": testloader}
        len_dict = {"train": len(tr_dataset), "valid": len(te_dataset)}

        subs = max(tr_dataset.sites) if ddloss == "sites" else max(tr_dataset.names)
        if mode == "single":
            model_ft = SingleBranchCNN(use_pretrained=True, subs=subs + 1)
        else:
            model_ft = MultiBranchCNN(use_pretrained=True, subs=subs + 1)
        model_ft = model_ft.to(device)

        # loss function with class weights
        r = tr_dataset.get_ratio()
        weights = torch.tensor([r, 1 / r]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model, _ = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            lens=len_dict,
        )
        # get_saliency(model, te_dataset)
        auc, acc, sns, spc, raw = test_model(model, testloader, raw)
        aucs.append(100 * auc)
        accs.append(100 * acc)
        snss.append(100 * sns)
        spcs.append(100 * spc)

    # log the mean metrics for each fold
    with open(logger, "a") as f:
        f.write("Fold " + str(fold))
        f.write("\nAUC: {:.1f}".format(np.mean(aucs)))
        f.write("\nAcc: {:.1f}".format(np.mean(accs)))
        f.write("\nSen: {:.1f}".format(np.mean(snss)))
        f.write("\nSpe: {:.1f}\n\n".format(np.mean(spcs)))

    return raw
