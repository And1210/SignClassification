import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score
from torch.optim import SGD

class TEST_MODULE(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(FER_TEST, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.bn3 = nn.BatchNorm2d(num_features=512)

        self.fc = nn.Linear(512, 7)

    def forward(self):
        x = self.input

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Testmodel(BaseModel):

    def __init__(self, configuration, in_channels=1, num_classes=7):
        super().__init__(configuration)

        self.model = TEST_MODULE(in_channels, num_classes)
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

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []

    def forward(self):
        x = self.input

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        self.output = x
        return x

    def compute_loss(self):
        self.loss_total = self.criterion_loss(self.output, self.label)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()
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
        metrics['accuracy'] = val_accuracy

        visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

def basenet(in_channels=1, num_classes=7):
    return FER_TESTmodel(in_channels, num_classes)


if __name__ == "__main__":
    net = TEST_MODULEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
