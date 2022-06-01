import torch.nn as nn, torch
from torchvision import models


class SingleBranchCNN(nn.Module):
    def __init__(self, mode, feature_extract, use_pretrained):
        super(SingleBranchCNN, self).__init__()
        self.mode = mode

        self.base_net = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(self.base_net, feature_extract)
        num_ftrs = self.base_net.classifier.in_features
        self.base_net.classifier = nn.Linear(num_ftrs, 128)

        self.cls = nn.Sequential(nn.LeakyReLU(), nn.Linear(128, 2))

    def forward(self, inp):
        out = self.base_net(inp[self.mode])
        return self.cls(out)


class MultiBranchCNN(nn.Module):
    def __init__(self, feature_extract, use_pretrained, branches=3):
        super(MultiBranchCNN, self).__init__()

        self.base_net = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(self.base_net, feature_extract)
        num_ftrs = self.base_net.classifier.in_features
        self.base_net.classifier = nn.Linear(num_ftrs, 128)
        self.base_net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 128))

        self.red_net = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(self.red_net, feature_extract)
        num_ftrs = self.red_net.classifier.in_features
        self.red_net.classifier = nn.Linear(num_ftrs, 128)
        self.red_net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 128))

        self.green_net = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(self.green_net, feature_extract)
        num_ftrs = self.green_net.classifier.in_features
        self.green_net.classifier = nn.Linear(num_ftrs, 128)
        self.green_net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 128))

        self.cls = nn.Sequential(nn.LeakyReLU(), nn.Linear(128 * branches, 2))

    def forward(self, inp):
        out1 = self.base_net(inp[0])
        out2 = self.red_net(inp[1])
        out3 = self.green_net(inp[2])
        outs = torch.cat((out1, out2, out3), dim=1)
        return out1, out2, out3, self.cls(outs)


def set_parameter_requires_grad(model, feature_extract):
    for i, child in enumerate(model.children()):
        for param in child.parameters():
            if feature_extract or i:
                param.requires_grad = True
            else:
                param.requires_grad = False
