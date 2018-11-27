import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import config
from torch.autograd import Variable


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=(1, 1), stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        # bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        # if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
        #     bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        # else:
        #     bottleneck_output = bn_function(*prev_features)
        concated_features = torch.cat(prev_features, 1)
        new_features = self.conv2(self.relu2(self.norm2(concated_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        init_features = init_features.detach().cpu().numpy()
        init_features = init_features[:, :, :config.max_len, :]
        init_features = torch.Tensor(init_features)
        features = [init_features.cuda()]
        for name, layer in self.named_children():
            new_features = layer(*features)
            new_features = new_features.detach().cpu().numpy()
            new_features = new_features[:, :, :config.max_len, :]   # 由于特征图的下面多了一行padding值，所以通过切片取前面max_len行特征
            new_features = torch.Tensor(new_features)
            features.append(new_features.cuda())
        return torch.cat(features, 1)   # 将所有生成的特征图拼接在一起，用于最后的全连接


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=16, block_config=(4, 1, 1), compression=0.5,
                 num_init_features=16, bn_size=4, drop_rate=0,
                 num_classes=config.num_classes, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        self.embed = nn.Embedding(config.vocab_size+2, config.embed_size)

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, num_init_features, kernel_size=(2, config.embed_size), stride=1, padding=(1, 0), bias=False)),
            ]))
            print("self.features:", self.features)
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config[:1]):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # if i != len(block_config) - 1:
            #     trans = _Transition(num_input_features=num_features,
            #                         num_output_features=int(num_features * compression))
            #     self.features.add_module('transition%d' % (i + 1), trans)
            #     num_features = int(num_features * compression)

        # Final batch norm
        # self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)   # 用于分类的全连接层，得到对每个类别标签的预测值

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        x = self.embed(Variable(torch.LongTensor(x.cpu().numpy()).cuda()))   # 将词向量嵌入样本序列中
        x = torch.unsqueeze(x, 1)   # 添加代表通道的维度（与TensorFlow不同，pytorch的通道维度是在第1个维度，也就是在代表batch的第0维之后，而TensorFlow是用最后一维度来表示通道）
        features = self.features(x)  # 用DenseNet的DenseBlock提取特征
        out = F.relu(features, inplace=True)
        out = F.max_pool2d(out, kernel_size=(out.size(2), 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out
