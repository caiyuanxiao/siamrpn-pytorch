from __future__ import absolute_import, division  #导入python未来支持的绝对引入、divison精确除法

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np                #numpy是python自带的数值扩展计算，可以存储和处理大型矩阵
import cv2
from collections import namedtuple     #collections模块提供了一些有用的集合类，可以根据需要选用。
                                       #用namedtuple可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用
from got10k.trackers import Tracker


class SiamRPN(nn.Module):               #nn.Module 是所有神经网络单元（neural network modules）的基类。
                                        #pytorch在nn.Module中，实现了__call__方法，而在__call__方法中调用了forward函数。

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()    #首先找到SiamRPN的父类，然后把类SiamRPN的对象self转换为父类的对象，然后“被转
                                           # 换”的父类对象调用自己的__init__函数
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(      #nn.Sequential表示一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算
                                           #图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            # conv1
            nn.Conv2d(3, 192, 11, 2),      #3通道，输出192通道，卷积核大小11，步长2
            nn.BatchNorm2d(192),           #对输出192通道的feature map批标准化
            nn.ReLU(inplace=True),        #inplace=True的意思是进行原地操作，
                                          #对上层网络传递下来的数据直接进行修改，好处就是可以节省运算内存，不用加储存变量
            nn.MaxPool2d(3, 2),           #卷积核大小3，步长2
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)  #通过CNN后，模板z的feature map通过一个卷积层升维，维数升为4k*512
                                                                      #其中512为CNN输出通道，k为anchor数量，4表示dx,dy,dw,dh
        self.conv_reg_x = nn.Conv2d(512, 512, 3)               #通过CNN后，搜索x的feature map不升维
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)  #通过CNN后，模板z的feature map通过一个卷积层升维，维数升为2k*512
                                                                     #其中512为CNN输出通道，k为anchor数量，2表示前景/背景分数
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, **self.learn(z))

    def learn(self, z):         #构建相关操作的核
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)    #使用升维后的模板作为reg分支的核
        kernel_cls = self.conv_cls_z(z)    #使用升维后的模板作为cls分支的核

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls        #相关操作的核构建完毕

    def inference(self, x, kernel_reg, kernel_cls):   #定义相关操作
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)
        
        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))    #reg分支进行相关
        out_cls = F.conv2d(x_cls, kernel_cls)                      #cls分支进行相关

        return out_reg, out_cls        #输出reg分支、cls分支的相关结果（响应图）


class TrackerSiamRPN(Tracker):    

    def __init__(self, net_path=None, **kargs):        #加*时，函数可接受任意多个参数，全部放入一个元组中
                                                       #加**时，函数接受参数时，返回为字典，参数形式应为键名=值名
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')  #定义device，其中需要注意的是“cuda:0”代表起始的device_id为0，
                                                           #如果直接是“cuda”，同样默认是从0开始。可以根据实际需要修改起始位置，如“cuda:1”
                                                           #一个device是一个对象，这个对象表示torch.sensor在哪个device上
                                                          #device对象包含device.type（cpu、cuda），还有一个可选择的设备序列号，如果该序列号
                                                          #没有声明，代表使用当前设备
                                                          #上述语句表示如果存在cuda则使用当前gpu运行，否则使用cpu运行

        # setup model
        self.net = SiamRPN()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))   #pytorch有两种模型保存方式。1保存整个神经网络
                                                                        #的结构信息和模型参数信息，save的对象是网络net。
                                                                        #2只保存神经网络的训练模型参数，save的对象是net.state_dict()
                                                                        #网络较大时，选择第二种节约时间
                                                                        #对应两种保存模型的方式，pytorch也有两种加载模型的方式
                                                                  #第一种保存方式，加载模型时通过torch.load('.pth')直接初始化新的神经网络对象
                                                                      #第二种保存方式，需要首先导入对应的网络，
                                                                      #再通过net.load_state_dict(torch.load('.pth'))完成模型参数的加载。
        self.net = self.net.to(self.device)

    def parse_args(self, **kargs):      #定义参数
        self.cfg = {
            'exemplar_sz': 127,       #输入模板尺寸127
            'instance_sz': 271,       #输入样本尺寸271
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],  #5种anchor长宽比
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():     #kargs.items()以列表的形式返回kargs的所有键值对，kargs为上面以字典形式输入的参数
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)  #建立一个命名元组，名字为GenericDict,元素为self.cfg.keys
                                                                           #cfg文件一般是程序运行的配置文件

    def init(self, image, box):
        image = np.asarray(image)       #将输入转换为矩阵格式

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([           
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)              #创建一个ndarry，数据类型为float32
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:   #np.prod()函数用来计算所有元素的乘积，对于有多个维度的数组可以指定轴，
                                                                         #如axis=1指定计算每一行的乘积。
            self.cfg = self.cfg._replace(instance_sz=287)      #改变样本尺寸为287，默认为271

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)

    def update(self, image):
        image = np.asarray(image)
        
        # search image
        instance_image = self._crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)              #构建搜索图像

        # classification and regression outputs
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)
        
        # offsets
        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        response = F.softmax(out_cls.permute(                      
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
                                                                     #.permute()将tensor的维度换位。三维的tensor维度序号分别为0,1,2
                                                                     #如一个tensor=[77,88,99],使用permute（2,1,0）转换为tensor=[99,88,77]
                                                                     #.view()是改变tensor的维数，比如一个2×3的tensor，使用view(3,2)变成
                                                                     #3×2的新tensor。view(2,-1)表示新的tensor行数=2，列数不确定
                                                                     #dim=0表示对每一列进行softmax，dim=1表示对每一行进行softmax
                  
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        
        # peak location
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz
                                                                        #np.argmax找出矩阵中最大值元素对应的索引值
                                                                        #对一维矩阵a=[3,1,2,4,6],最大值元素为6，对应索引为4，np.argmax(a)=4
                                                                        #对二维矩阵b=[[1,5,3];[9,6,2]],np.argmax(b,axis=1)表示按行进行查找
                                                                        #第一行最大值5，对应索引2,；第二行最大值9，对应索引0，输出[2,0]
                                                                        #np.argmax(b,axis=0)表示按列进行查找
                                                                  #第一列最大值9，对应索引1；第二列最大值6，对应索引1；第三列最大值3，对应索引0
                                                                   #输出[1,1,0]

        # update center
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])   #np.clip(a, a_min, a_max)对原数组进行剪切。a表示原数组，
                                                                  #a_min表示限定的最小值 也可以是数组 如果为数组时 shape必须和a一样
                                                                  #a_max表示限定的最大值 也可以是数组 shape和a一样

        # update scale
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)   
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz           #np.sum(a),将a中元素全部相加
                                                                  #np.sqrt(a),对a求平方根

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))  #np.tile()表示把数组沿各个方向复制
                                                                  #如a=np.array([1,2];[3,4])。np.tile(a,2)表示把a沿行方向复制两倍
                                                            #变成([1,2,1,2];[3,4,3,4])。np.tile(a,(2,1))表示把a沿列方向复制2倍，行方向复制1倍
                                                            #变成([1,2];[3,4];[1,2];[3,4])

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))    #meshgrid函数用两个坐标轴上的点在平面上画网格
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)
        
        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)                                  #round() 方法返回浮点数x的四舍五入值
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((                                  #np.concatenate表示对多个数组进行拼接
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(                          #cv2.copyMakeBorder()用来给图片添加边框，即padding。
                                                                 #参数cv2.copyMakeBorder(image,top,bottom,left,right)
                                                                 #其中image表示原图，top,bottom,left,right表示上下左右分别填充的像素数
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)           #cv2.BORDER_CONSTANT表示固定值填充，边框都填充成一个固定值

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch
