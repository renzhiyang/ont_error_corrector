import torch
import math
import params
import timm
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.optim as optim
from torch import nn, Tensor
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader

class CNN_6layer(nn.Module):
    def __init__(self):
        super(CNN_6layer, self).__init__()
        # 1x101x30  1:read bases,  no padding here, ont-hot encodding
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7,7), stride=(1,1), padding=(3,3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5,3), stride=(1,1), padding=(2,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#model = cnn_2layer_model()
#summary(model, input_size=(4, 101, 30))

# RestNet with flexible kernel size
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers=[2,2,2,2], num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers with custom kernel sizes
        #self.layer1 = self._make_layer(block, 64, layers[0], kernel_size=(5, 5))
        #self.layer2 = self._make_layer(block, 128, layers[1], kernel_size=(5, 3), stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], kernel_size=(3, 3))
        #self.layer4 = self._make_layer(block, 512, layers[3], kernel_size=(3, 1), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, kernel_size, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock_constant_kernel(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_constant_kernel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_constant_kernel(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet_constant_kernel, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Example Data Preparation (You'll need to adapt this to your data specifics)
#vocab_size = 5  # For A, C, G, T, others
#num_features = 9 # for each base, embedding to [1,9] with 9 features
#max_length = 101 # the number of bases in one sentence, window width
#num_sequences = 30  # the number of reads, window height
class CustomDistilBert(nn.Module):
    def __init__(self, vocab_size, num_features, max_length,num_sequences, num_classes=2):
        super(CustomDistilBert, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, num_features)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=num_sequences*num_features, nhead=3)
        self.pooler = nn.Linear(num_sequences*num_features, num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.max_length = max_length

    def forward(self, input_ids):
        # Embedding layer
        x = input_ids.permute(0, 2, 1).long()
        x = self.embeddings(x)
        
        x_0_d = x.size(0)
        x_1_d = x.size(1)
        #print(x_0_d, x_1_d)
        x = x.contiguous().view(x_0_d, x_1_d, -1)
        
        # Transformer blocks
        for _ in range(1):  # DistilBERT typically has 6 layers
            x = self.transformer_block(x)
        
        # Pooling
        pooled_output = self.pooler(x.mean(dim=1))

        # Classifier
        logits = self.classifier(pooled_output)
        return logits
