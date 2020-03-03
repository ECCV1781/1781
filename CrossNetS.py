import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

class Restorator(torch.nn.Module):
    def __init__(self):
        super(Restorator, self).__init__()

        def conv2(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, padding=2, stride=stride),
                nn.ReLU(inplace=True)
            )

        def deconv2(in_planes, out_planes, interp = True):
            if True :
                return nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True)
            )
            else:
                return nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

        self.conv0 = conv2(4, 64)
        self.conv1 = conv2(64, 64)
        self.conv2 = conv2(64, 64, 2)
        self.conv3 = conv2(64, 64, 2)
        self.conv4 = conv2(64, 64, 2)

        self.deconv4 = deconv2(64 * 2, 64)
        self.deconv3 = deconv2(64 * 2, 64)
        self.deconv2 = deconv2(64 * 2, 64)
        self.deconv1 = conv2(64 * 2, 64)
        self.recon = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2, stride=1)
        )
        

        self.resample1 = Backward()
        self.resample2 = Backward()
        self.resample3 = Backward()
        self.resample4 = Backward()

    def forward(self, x, color, color_q, flow):
        #depth color flow
        # input
        out = self.conv0(torch.cat([x, color], 1))
        out1 = self.conv1(out)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        Lwarp_out1 = self.resample1(out4, flow[0])
        Lwarp_out2 = self.resample2(out3, flow[1])
        Lwarp_out3 = self.resample3(out2, flow[2])
        Lwarp_out4 = self.resample4(out1, flow[3])

        # Deconv
        out5 = self.deconv4(torch.cat([out4, Lwarp_out1], 1))
        out6 = self.deconv3(torch.cat([out5, Lwarp_out2], 1))
        out7 = self.deconv2(torch.cat([out6, Lwarp_out3], 1))
        out8 = self.deconv1(torch.cat([out7, Lwarp_out4], 1))
        out9 = self.recon(out8)

        return out9


class FlowNetS(nn.Module):
    def __init__(self, input_channels=6, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 32, kernel_size=7)
        self.conv1_1 = conv(self.batchNorm, 32, 32, kernel_size=5)

        self.conv2 = conv(self.batchNorm, 32, 64, kernel_size=5, stride=2)
        self.conv2_1 = conv(self.batchNorm, 64, 64, kernel_size=5)

        self.conv3 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 128, 128)

        self.conv4 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv4_1 = conv(self.batchNorm, 256, 256)

        self.conv5 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)

        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(514, 128)
        self.deconv2 = deconv(258, 64)
        self.deconv1 = deconv(130, 32)
        self.deconv0 = deconv(66, 16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(514)
        self.predict_flow3 = predict_flow(258)
        self.predict_flow2 = predict_flow(130)
        self.predict_flow1 = predict_flow(66)


        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)


    def forward(self, x):
        out_conv1 = self.conv1_1(self.conv1(x))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
      
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)

        return [flow4, flow3, flow2, flow1]


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=False)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=False)
    )
    

class Backward(torch.nn.Module):
    def __init__(self):
        super(Backward, self).__init__()
    # end

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.flow = FlowNetS(input_channels=7)        
        self.restore = Restorator()

    def forward(self, depth, color, color_q, isTrain=True):
        
        flow = self.flow(torch.cat([color_q, color, depth], 1))  

        res = self.restore(depth, color, color_q, flow)        

        if isTrain:
            return depth + res
        else:
            return depth + res, flow

