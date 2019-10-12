import mxnet as mx
from mxnet import nd,gluon
from mxnet.gluon import nn,HybridBlock
import matplotlib.pyplot as plt
import matplotlib.image as mp


import os
import pathlib
import numpy as np
from mxnet.gluon.data import Dataset,RecordFileDataset
from datetime import datetime
import time
import logging
from mxnet import autograd,image,nd,recordio
from mxnet import init,initializer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision.transforms import ToTensor
from mxboard import SummaryWriter
import cv2
import numpy as np
epochs=10000
batch_size=128
ctx=mx.gpu(2)
# ctx=[mx.gpu(i) for i in [2,3]]
lr=0.001
alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.()载自重换长容积_'
print(len(alphabet))
num_class=len(alphabet)
log_dir='/home/cumt306/zhouyi/txt_gan/logs'

target_wd = 128
target_ht=32
img_list=[]
num=0
def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)

def save_checkpoint(net, epoch,filename, is_best=False):
    # “”“Save Checkpoint”""
    directory = "model/%s/"% (epoch)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename+'.params'
    net.save_parameters(filename)

class ImageDataset(Dataset):
    def __init__(self, data_txt: str, data_shape: tuple, img_channel: int, num_label: int,
                 alphabet: str, phase: str = 'train'):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数,应该和网络最终输出的序列宽度一样
        :param alphabet: 字母表
        """
        super(ImageDataset, self).__init__()
        assert phase in ['train', 'test']

        self.data_list = []
        with open(data_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                img_path = pathlib.Path(line[0])
                if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                    self.data_list.append((line[0], line[1]))
        self.img_h = data_shape[0]
        self.img_w = data_shape[1]
        self.img_channel = img_channel
        self.num_label = num_label
        self.alphabet = alphabet
        self.phase = phase
        self.label_dict = {}
        for i, char in enumerate(self.alphabet):
            self.label_dict[char] = i

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        label = label.replace(' ', '')
        try:
            label = self.label_enocder(label)
        except Exception as e:
            print(img_path, label)
        img = self.pre_processing(img_path)
        return img, label

    def __len__(self):
        return len(self.data_list)

    def label_enocder(self, label):
        """
        对label进行处理，将输入的label字符串转换成在字母表中的索引
        :param label: label字符串
        :return: 索引列表
        """
        tmp_label = nd.zeros(self.num_label, dtype=np.float32) - 1
        for i, ch in enumerate(label):
            tmp_label[i] = self.label_dict[ch]

        return tmp_label


    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度

        :param img_path: 图片地址
        :return:
        """
        data_augment = False
        # if self.phase == 'train' and np.random.rand() > 0.5:
        #     data_augment = True
        if data_augment:
            img_h = 40
            img_w = 340
        else:
            img_h = self.img_h
            img_w = self.img_w

        img = image.imdecode(open(img_path, 'rb').read(), 1 if self.img_channel == 3 else 0)
        h, w = img.shape[:2]
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)

        ################
        # img = image.imresize(img, w=self.img_w, h=self.img_h)
        if new_w < img_w:
            img = image.imresize(img, w=new_w, h=img_h)
            step = nd.zeros((img_h, img_w - new_w, self.img_channel), dtype=img.dtype)
            img = nd.concat(img, step, dim=1)
        else:
            img = image.imresize(img, w=img_w, h=img_h)

        # if data_augment:
        #     img, _ = image.random_crop(img, (self.img_w, self.img_h))
        return img

def _infer_weight_shape(op_name, data_shape, kwargs):
    op = getattr(mx.symbol, op_name)
    sym = op(mx.symbol.var('data', shape=data_shape), **kwargs)
    return sym.infer_shape_partial()[0]

class _ConvSN(nn.HybridBlock):

    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='Convolution', adj=None, prefix=None, params=None,
                 power_iterations=1, epsilon=1e-8):
        super(_ConvSN, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(strides, mx.base.numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, mx.base.numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, mx.base.numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._op_name = op_name
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}
            if adj is not None:
                self._kwargs['adj'] = adj

            dshape = [0]*(len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels
            wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
            self.weight = self.params.get('weight_spectral_norm', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)

            self.u = self.params.get('u_spectral_norm', init=mx.init.Normal(), shape=(1, wshapes[1][0]),
                                     grad_req='null', differentiable=False)
            self.sigma = self.params.get('sigma_spectral_norm', init=mx.init.Constant(1.0), shape=(1,),
                                         grad_req='null', differentiable=False)
            self._power_iterations = power_iterations
            self._eps = epsilon

            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, u, sigma, bias=None):

        # sigma = F.maximum(sigma, self._eps)
        weight = F.broadcast_div(weight, sigma)
        # if isinstance(sigma, mx.nd.NDArray):
        #     if self.prefix.startswith('4_conv_1'):
        #         print('forward: {}'.format(sigma.asscalar()))

        if bias is None:
            act = getattr(F, self._op_name)(x, weight, name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, weight, bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)

def image_processing(img):
    img_aft=img.copy()
    h_start=int(np.random.randint(0,4))*5
    w_start=int(np.random.randint(0,16))*5
    gray=int(np.random.randint(0,256))
    #print(w_start)
    width=6
    hight=6
    # image_aft=img
    img_aft[:,:,h_start:(h_start+hight),w_start:(w_start+width)]=gray
    return img_aft

class Conv2DSN(_ConvSN):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, mx.base.numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(Conv2DSN, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)

class Conv2DTransposeSN(_ConvSN):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 output_padding=(0, 0), dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, mx.base.numeric_types):
            kernel_size = (kernel_size,)*2
        if isinstance(output_padding, mx.base.numeric_types):
            output_padding = (output_padding,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        assert len(output_padding) == 2, "output_padding must be a number or a list of 2 ints"
        super(Conv2DTransposeSN, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer,
            bias_initializer, op_name='Deconvolution', adj=output_padding, **kwargs)
        self.outpad = output_padding

class DenseSN(nn.HybridBlock):
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, power_iterations=1, epsilon=1e-8, **kwargs):
        super(DenseSN, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight_spectral_norm', shape=(units, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)
            self.u = self.params.get('u_spectral_norm', init=mx.init.Normal(), shape=(1, units),
                                     grad_req='null', differentiable=False)
            self.sigma = self.params.get('sigma_spectral_norm', init=mx.init.Constant(1.0), shape=(1,),
                                         grad_req='null', differentiable=False)
            self._power_iterations = power_iterations
            self._eps = epsilon

            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, u, sigma, bias=None):
        # sigma = F.maximum(sigma, self._eps)
        weight = F.broadcast_div(weight, sigma)

        act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))

class Encorder(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Encorder, self).__init__(**kwargs)
        with self.name_scope():
            self.net=nn.HybridSequential()
            self.net.add(Conv2DSN(channels=64,kernel_size=3,strides=1,padding=1,in_channels=3))#()
            self.net.add(Conv2DSN(channels=64, kernel_size=2, strides=2, padding=0, in_channels=64))#(n,64,16,64)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.LeakyReLU(0.2))
            self.net.add(Conv2DSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=64))
            self.net.add(Conv2DSN(channels=128, kernel_size=2, strides=2, padding=0, in_channels=128))
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.LeakyReLU(0.2))
            self.net.add(Conv2DSN(channels=256, kernel_size=3, strides=1, padding=1, in_channels=128))
            self.net.add(Conv2DSN(channels=256, kernel_size=(2,1), strides=(2,1), padding=0, in_channels=256))#(n,256,4,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.LeakyReLU(0.2))
            self.net.add(Conv2DSN(channels=512, kernel_size=(2, 1), strides=(2, 1), padding=0, in_channels=256))  # (n,256,2,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.LeakyReLU(0.2))
            self.net.add(Conv2DSN(channels=512, kernel_size=(2, 1), strides=(2, 1), padding=0, in_channels=512))
            # self.net.add(Conv2DSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=3))
            # self.net.add(Conv2DSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=3))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)

class Decorder(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Decorder, self).__init__(**kwargs)
        with self.name_scope():
            self.net=nn.HybridSequential()
            self.net.add(Conv2DTransposeSN(channels=256,kernel_size=(2,1),strides=(2,1),padding=0,in_channels=512))#(n,256,2,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DTransposeSN(channels=256, kernel_size=(2, 1), strides=(2, 1), padding=0,in_channels=256))  # (n,256,4,32)
            self.net.add(Conv2DTransposeSN(channels=256, kernel_size=3, strides=1, padding=1, in_channels=256))# (n,256,4,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DTransposeSN(channels=256, kernel_size=(2, 1), strides=(2, 1), padding=0,in_channels=256))  # (n,256,8,32)
            self.net.add(Conv2DTransposeSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=256))  # (n,128,8,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DTransposeSN(channels=128, kernel_size=2, strides=2, padding=0,in_channels=128))  # (n,128,8,32)
            self.net.add(Conv2DTransposeSN(channels=64, kernel_size=3, strides=1, padding=1, in_channels=128))  # (n,64,16,64)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DTransposeSN(channels=64, kernel_size=2, strides=2, padding=0, in_channels=64))  # (n,128,8,32)
            self.net.add(Conv2DTransposeSN(channels=3, kernel_size=3, strides=1, padding=1, in_channels=64))  # (n,64,16,64)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('tanh'))


    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)

class Discriminator(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.net=nn.HybridSequential()
            self.net.add(Conv2DSN(channels=64, kernel_size=2, strides=2, padding=0, in_channels=3))#(n,64,16,64)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DSN(channels=128, kernel_size=2, strides=2, padding=0, in_channels=64))
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DSN(channels=256, kernel_size=2, strides=2, padding=0, in_channels=128))#(n,256,4,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DSN(channels=512, kernel_size=2, strides=2, padding=0, in_channels=256))  # (n,256,2,32)
            self.net.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
            self.net.add(nn.Activation('relu'))
            self.net.add(Conv2DSN(channels=512, kernel_size=2, strides=2, padding=0, in_channels=512))
            self.net.add(DenseSN(1,in_units=512*4))
            self.net.add(nn.Activation('tanh'))
            # self.net.add(Conv2DSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=3))
            # self.net.add(Conv2DSN(channels=128, kernel_size=3, strides=1, padding=1, in_channels=3))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)

class BidirectionalLSTM(HybridBlock):
    def __init__(self, hidden_size, num_layers, nOut):
        super(BidirectionalLSTM, self).__init__()
        with self.name_scope():
            self.rnn = mx.gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=True, layout='NTC')
            # self.fc = nn.Dense(units=nOut, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rnn(x)
        # x = self.fc(x)  # [T * b, nOut]
        return x

class RNN_Block(nn.HybridBlock):
    def __init__(self, n_class, hidden_size=256, num_layers=1,**kwargs):
        super(RNN_Block, self).__init__(**kwargs)
        with self.name_scope():
            self.LSTM_Model=nn.HybridSequential()
            self.LSTM_Model.add(BidirectionalLSTM(hidden_size,num_layers,hidden_size*2))
            self.LSTM_Model.add(BidirectionalLSTM(hidden_size, num_layers, n_class))
            self.LSTM_Model.add(nn.Dense(units=n_class,flatten=False))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = x.squeeze(axis=2)
        x = x.transpose((0, 2, 1))  # (NTC)(batch, width, channel)
        x = self.LSTM_Model(x)
        return x

def accuracy(predictions, labels, alphabet):
    predictions = predictions.softmax().topk(axis=2).asnumpy()
    zipped = zip(decode(predictions, alphabet), decode(labels.asnumpy(), alphabet))
    n_correct = 0
    for pred, target in zipped:
        if pred == target:
            n_correct += 1
    return n_correct

def evaluate_accuracy(net1,net2, dataloader, ctx, alphabet):
    metric = 0
    j=0
    for i, (data, label) in enumerate(dataloader):
        i=j+1
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net2(net1(data))
        metric += accuracy(output, label, alphabet)
    return metric

def decode(preds, alphabet, raw=False):
    results = []
    alphabet_size = len(alphabet)
    for word in preds:
        if raw:
            results.append(''.join([alphabet[int(i)] for i in word]))
        else:
            result = []
            for i, index in enumerate(word):


                if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                    continue
                if index == -1 or index >= alphabet_size - 1:
                    continue
                else:
                    result.append(alphabet[int(index)])
            results.append(''.join(result))
    return results

def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger

logger = setup_logger(os.path.join(log_dir, 'train_log'))

Generator_Encorder_net=nn.HybridSequential()
Generator_Encorder_net.add(Encorder())
Generator_Encorder_net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)

Generator_Decorder_net=nn.HybridSequential()
Generator_Decorder_net.add(Decorder())
Generator_Decorder_net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)

Discriminator_net=nn.HybridSequential()
Discriminator_net.add(Discriminator())
Discriminator_net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)

RNN_Block_net=nn.HybridSequential()
RNN_Block_net.add(RNN_Block(n_class=num_class))
RNN_Block_net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)


GE_trainer = gluon.Trainer(Generator_Encorder_net.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.9,'beta2': 0.999})
GD_trainer = gluon.Trainer(Generator_Decorder_net.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.9,'beta2': 0.999})
D_trainer = gluon.Trainer(Discriminator_net.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.9,'beta2': 0.999})
RNN_trainer = gluon.Trainer(RNN_Block_net.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.9,'beta2': 0.999})


#######################################################
######    dataset path    ######
dataset = ImageDataset('/home/cumt306/zhouyi/dataset/Train.txt', (32, 128), 3, 32, alphabet)
data_loader = DataLoader(dataset.transform_first(ToTensor()), batch_size=batch_size, shuffle=True, num_workers=12)
val_dataset = ImageDataset('/home/cumt306/zhouyi/dataset/Val.txt', (32, 128), 3, 32, alphabet)
val_data_loader = DataLoader(dataset.transform_first(ToTensor()), batch_size=batch_size, shuffle=True, num_workers=12)
test_dataset = ImageDataset('/home/cumt306/zhouyi/dataset/Test.txt', (32, 128), 3, 32, alphabet)
test_data_loader = DataLoader(test_dataset.transform_first(ToTensor()), batch_size=batch_size, shuffle=True, num_workers=12)
#######################################################



stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.DEBUG)
GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
ctc_loss = gluon.loss.CTCLoss(weight=0.2)
L1_loss=gluon.loss.L1Loss()
sw = SummaryWriter(log_dir)
global_step = 0

for epoch in range(epochs):

    loss = .0
    train_acc = .0
    tick = time.time()

    for i, (img, label) in enumerate(data_loader):
        global_step = global_step +1
        img = img.as_in_context(ctx)
        label = label.as_in_context(ctx)
        img_processed=image_processing(img)
        real_label = nd.ones((img.shape[0],), ctx=ctx)
        fake_label = nd.zeros((img.shape[0],), ctx=ctx)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        with autograd.record():
            # train with real image
            output = Discriminator_net(img).reshape((-1, 1))
            errD_real = GAN_loss(output, real_label)
            metric.update([real_label,], [output,])
            # train with fake image
            fake_Sequential = Generator_Encorder_net(img_processed)
            fake_image = Generator_Decorder_net(fake_Sequential)
            output=Discriminator_net(fake_image).reshape((-1, 1))
            errD_fake = GAN_loss(output, fake_label)
            errD = errD_real + errD_fake
            metric.update([fake_label,], [output,])
        errD.backward()

        D_trainer.step(img.shape[0])
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake_Sequential = Generator_Encorder_net(img_processed)
            fake_image = Generator_Decorder_net(fake_Sequential)
            output = Discriminator_net(fake_image).reshape((-1, 1))
            errG_norm = GAN_loss(output, real_label)

            Gen_Sequential=Generator_Encorder_net(fake_image)

            errG_Seq=L1_loss(Gen_Sequential,fake_Sequential)

            errG_img=L1_loss(fake_image,img)
            errG=errG_norm+0.4*errG_Seq+0.4*errG_img
        errG.backward()

        GE_trainer.step(img.shape[0])
        GD_trainer.step(img.shape[0])
        ############################
        # (3) Update RNN block: CTC loss
        ###########################

        Sequantial=Generator_Encorder_net(img)
        with autograd.record():
            output=RNN_Block_net(Sequantial)
            loss_ctc=ctc_loss(output, label)
            loss_ctc = (label != -1).sum(axis=1) * loss_ctc
        loss_ctc.backward()

        RNN_trainer.step(img.shape[0])

        acc = accuracy(output, label,alphabet)
        train_acc += acc
        name, GAN_D_acc = metric.get()

        if (i + 1) % 20 == 0:

            acc /= len(label)
            batch_time = time.time() - tick
            logger.info(
                '[{}/{}],step: {}, GAN D loss: {:.3f}, GAN G loss: {:.3f} , ctc loss: {:.4f},acc: {:.4f},GAN D acc: {:.4f}'
                .format(epoch, epochs, i,nd.mean(errD).asscalar(),nd.mean(errG).asscalar(),nd.mean(loss_ctc).asscalar(), acc,GAN_D_acc))
            sw.add_scalar(tag='GAN D loss', value=nd.mean(errD).asscalar(), global_step=global_step)
            sw.add_scalar(tag='GAN G loss', value=nd.mean(errG).asscalar(), global_step=global_step)
            sw.add_scalar(tag='ctc loss', value=nd.mean(loss_ctc).asscalar(), global_step=global_step)
            sw.add_scalar(tag='accuracy_per_step', value=acc, global_step=global_step)
            sw.add_scalar(tag='GAN_D_acc', value=GAN_D_acc, global_step=global_step)
            loss = .0
            tick = time.time()
            nd.waitall()
            ################################
    if (epoch)%20==0:
        save_checkpoint(Generator_Encorder_net,epoch=epoch,filename='Generator_Encorder_net')
        save_checkpoint(Generator_Decorder_net, epoch=epoch, filename='Generator_Decorder_net')
        save_checkpoint(Discriminator_net, epoch=epoch, filename='Discriminator_net')
        save_checkpoint(RNN_Block_net, epoch=epoch, filename='RNN_Block_net')

    if epoch %10==0:
        validation_accuracy = evaluate_accuracy(Generator_Encorder_net, RNN_Block_net, val_data_loader, ctx,
                                                alphabet) / val_dataset.__len__()
        logger.info('[{}/{}],validation_accuracy: {:.4f}'.format(epoch, epochs, validation_accuracy))
        sw.add_scalar(tag='validation_accuracy', value=validation_accuracy, global_step=epoch)

    train_acc /= dataset.__len__()
    sw.add_scalar(tag='Train_accuracy', value=train_acc, global_step=epoch)

sw.close()
test_accuracy = evaluate_accuracy(Generator_Encorder_net, RNN_Block_net, test_data_loader, ctx,
                                                alphabet) / test_dataset.__len__()