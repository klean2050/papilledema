import torch.nn as nn, torch
from torchvision import models
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(_, x):
        return x.view_as(x)

    @staticmethod
    def backward(_, grad_output):
        return grad_output.neg()


class Sub_Adaptor(nn.Module):
    def __init__(self, num_feats, classes):
        super(Sub_Adaptor, self).__init__()
        self.linear = nn.Linear(num_feats, classes)
        self.relu = nn.LeakyReLU()

    def GRL(self, x):
        return GradReverse.apply(x)

    def forward(self, embed):
        return self.linear(self.GRL(embed))


class SingleBranchCNN(nn.Module):
    def __init__(self, use_pretrained, subs):
        super(SingleBranchCNN, self).__init__()

        self.base_net = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(self.base_net)
        num_ftrs = self.base_net.classifier.in_features
        self.base_net.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 128)
        )
        self.cls = nn.Sequential(nn.LeakyReLU(), nn.Linear(128, 2))
        self.sub = Sub_Adaptor(128, subs)

    def forward(self, inp):
        rand_ch = torch.randint(1, 3, size=(1,))
        out = self.base_net(inp[rand_ch[0]])
        return self.cls(out), self.sub(out)


class MultiBranchCNN(nn.Module):
    def __init__(self, use_pretrained, subs):
        super(MultiBranchCNN, self).__init__()

        self.base_net = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(self.base_net)
        num_ftrs = self.base_net.classifier.in_features
        self.base_net.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 128)
        )

        self.red_net = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(self.red_net)
        num_ftrs = self.red_net.classifier.in_features
        self.red_net.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 128)
        )

        self.green_net = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(self.green_net)
        num_ftrs = self.green_net.classifier.in_features
        self.green_net.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 128)
        )

        self.cls = nn.Sequential(nn.LeakyReLU(), nn.Linear(128, 2))
        self.sub = Sub_Adaptor(3 * 128, subs)

    def forward(self, inp):
        out1 = self.base_net(inp[0])
        out2 = self.red_net(inp[1])
        out3 = self.green_net(inp[2])
        concat = torch.stack([out1, out2, out3])

        dd_out = self.sub(concat.view(-1, 128 * 3))
        preds = self.cls(torch.max(concat, dim=0)[0])
        return preds, dd_out


def set_parameter_requires_grad(model):
    for child in model.children():
        for name, param in child.named_parameters():
            param.requires_grad = False
            if name == "denseblock4.denselayer16.norm1.weight":
                break
        break
