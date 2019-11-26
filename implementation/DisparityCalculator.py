from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
        nn.BatchNorm3d(out_planes)
    )

class Hourglass(nn.Module):
    def __init__(self, inplanes):
        super(Hourglass, self).__init__()

        self.convolution_1 = nn.Sequential(
            convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )

        self.convolution_2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.convolution_3 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )

        self.convolution_4 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
            nn.ReLU(inplace=True)
        )

        self.convolution_5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes * 2)
        )

        self.convolution_6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes)
        )
    
    def forward(self, input, presqu, postsqu):
        output = self.convolution_1(input)
        pre = self.convolution_2(output)

        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        output = self.convolution_3(pre)
        output = self.convolution_4(output)

        if presqu is not None:
            post = F.relu(self.convolution_5(output) + presqu, inplace=True)
        else:
            post = F.relu(self.convolution_5(output) + pre, inplace=True)
        
        output = self.convolution_6(post)

        return output, pre, post

class StackedHourglass(nn.Module):
    def __init__(self, max_disparity):
        super(StackedHourglass, self).__init__()
        self.max_disparity = max_disparity

        self.feature_extraction = feature_extraction()

        self.downsampling_0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.downsampling_1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1)
        )

        self.downsampling_2 = Hourglass(32)

        self.downsampling_3 = Hourglass(32)

        self.downsampling_4 = Hourglass(32)

        self.classification_1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.Relu(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classification_2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.Relu(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        
        self.classification_3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.Relu(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, left_image, right_image):
        refined_image_features = self.feature_extraction(left)
        targeting_features = self.feature_extraction(right)

        #matching
        cost = Variable(
            torch.FloatTensor(
                refined_image_features.size()[0],
                refined_image_features.size()[1] * 2,
                int(self.max_disparity / 4),
                refined_image_features.size()[2],
                refined_image_features.size()[3]
            ).zero_()
        ).cuda()

        for i in range(int(self.max_disparity)):
            if i > 0:
                cost[:, :refinined_image_features.size()[1], i, :, i:] = refined_image_features[:,:,:,i:]
                cost[:, refinined_image_features.size()[1]:, i, :, i:] = targeting_features[:,:,:,:-i]
            else:
                cost[:, :refined_image_features.size()[1], i, :, :] = refined_image_features
                cost[:, refined_image_features.size()[1]:, i, :, :] = targeting_features
        
        cost = cost.contiguous()

        cost0 = self.downsampling_0(cost)
        cost0 = self.downsampling_1(cost0) + cost0
        
        out1, pre1, post1 = self.downsampling_2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.downsampling_3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.downsampling_4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classification_1(out1)
        cost2 = self.classification_2(out2) + cost1
        cost3 = self.classification_3(out3) + cost2

        cost1 = F.upsample(cost1, [
            self.max_disparity,
            left.size()[2], left.size()[3],
            mode='trilinear'
        ])

        cost2 = F.upsample(cost2, [
            self.max_disparity,
            left.size()[2], left.size()[3],
            mode='trilinear'
        ])

        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparityregression(self.max_disparity)(pred1)

        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparityregression(self.max_disparity)(pred2)

        cost3 = F.upsample(cost3, [
            self.max_disparity,
            left.size()[2], left.size()[3],
            mode='trilinear'
        ])
        pred3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)

        pred3 = disparityregression(self.max_disparity)(pred3)

        return pred3

class DisparityCalculator():
    def __init__(self, max_disparity, model_path):
        super().__init__()
        self.max_disparity = max_disparity
        
        model = Hourglass(max_disparity)
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])

        # model.eval()

        self.model = model

    def calculate_disparity(self, left_image, right_image):
        image_left = torch.FloatTensor(left_image).cuda()
        image_right = torch.FloatTensor(right_image).cuda()

        image_left, image_right = Variable(image_left), Variable(image_right)

        with torch.no_grad():
            calculated_disparity = self.model(image_left, image_right)

        calculated_disparity = torch.squeeze(calculated_disparity)
        output_disparity = calculated_disparity.data.cpu().numpy()

        return output_disparity    