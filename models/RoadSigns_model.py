import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score
from collections import OrderedDict


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUnit, self).__init__()
        width = int(out_channels / 4)

        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # for downsample
        self._downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self._downsample(identity)
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass

class RoadSign(nn.Module):
    """basenet for RoadSign"""
    def __init__(self, in_channels=1, num_classes=47):
        super(RoadSign, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualUnit(in_channels=64, out_channels=64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=3,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.res2 = ResidualUnit(in_channels=128, out_channels=128)

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        self.fc = nn.Linear(25088, num_classes)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.tanh(x)
        return x

class RoadSignsmodel(BaseModel):
    """basenet for road sign detection"""

    def __init__(self, configuration, in_channels=1, num_classes=47):
        super().__init__(configuration)

        self.model = RoadSign(in_channels, num_classes)
        self.model.cuda()

        self.criterion_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=configuration['lr'],
            momentum=configuration['momentum'],
            weight_decay=configuration['weight_decay']
        )
        self.optimizers = [self.optimizer]

        self.loss_names = ['total']
        self.network_names = ['model']

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []

    def forward(self):
        x = self.input
        self.output = self.model.forward(x)
        return self.output

    def compute_loss(self):
        self.loss_total = self.criterion_loss(self.output, self.label)

    def optimize_parameters(self):
        self.loss_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

def basenet(in_channels=1, num_classes=4):
    return BaseNet(in_channels, num_classes)


if __name__ == "__main__":
    net = BaseNet().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
