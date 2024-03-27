import torch
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):

  def __init__(self, init_channel, next_channel):
    super(UNet, self).__init__()

    self.conv1 = nn.Conv2d(init_channel, next_channel, 3, padding='same')
    self.ReLU = nn.ReLU()
    self.conv2 = nn.Conv2d(next_channel, next_channel, 3, padding='same')
    self.maxPool = nn.MaxPool2d(2, 2, ceil_mode=True)
    self.conv3 = nn.Conv2d(next_channel, 2 * next_channel, 3, padding='same')
    self.conv4 = nn.Conv2d(2 * next_channel, 2 * next_channel, 3, padding='same')
    self.conv5 = nn.Conv2d(2 * next_channel, 4 * next_channel, 3, padding='same')
    self.conv6 = nn.Conv2d(4 * next_channel, 4 * next_channel, 3, padding='same')
    self.conv7 = nn.Conv2d(4 * next_channel, 8 * next_channel, 3, padding='same')
    self.conv8 = nn.Conv2d(8 * next_channel, 8 * next_channel, 3, padding='same')
    self.conv9 = nn.Conv2d(8 * next_channel, 16 * next_channel, 3, padding='same')
    self.conv10 = nn.Conv2d(16 * next_channel, 16 * next_channel, 3, padding='same')
    self.upsample = nn.Upsample(scale_factor=2)
    self.conv11 = nn.Conv2d(16 * next_channel, 8 * next_channel, 2, padding='same')

    #Apparently convtranspose2d also more or less does the same as upsample+conv2d.
    #Would be interesting to explore this too perhaps. But the paper says
    #upsample+conv, so I'll stick to this for now. Also, apparently, using transposed
    #convolution could have "checkboard artifacts" as compared to upsample+conv:
    #https://distill.pub/2016/deconv-checkerboard/

    #self.conv11 = nn.ConvTranspose2d(16 * next_channel, 8 * next_channel, 2, 2)

    self.conv12 = nn.Conv2d(16 * next_channel, 8 * next_channel, 3, padding='same')
    self.conv13 = nn.Conv2d(8 * next_channel, 8 * next_channel, 3, padding='same')
    self.conv14 = nn.Conv2d(8 * next_channel, 4 * next_channel, 2, padding='same')
    self.conv15 = nn.Conv2d(8 * next_channel, 4 * next_channel, 3, padding='same')
    self.conv16 = nn.Conv2d(4 * next_channel, 4 * next_channel, 3, padding='same')
    self.conv17 = nn.Conv2d(4 * next_channel, 2 * next_channel, 2, padding='same')
    self.conv18 = nn.Conv2d(4 * next_channel, 2 * next_channel, 3, padding='same')
    self.conv19 = nn.Conv2d(2 * next_channel, 2 * next_channel, 3, padding='same')
    self.conv20 = nn.Conv2d(2 * next_channel, next_channel, 2, padding='same')
    self.conv21 = nn.Conv2d(2 * next_channel, next_channel, 3, padding='same')
    self.conv22 = nn.Conv2d(next_channel, next_channel, 3, padding='same')
    self.conv23 = nn.Conv2d(next_channel, 1, 1)

    self.activation = nn.Sigmoid()

    # self.initialize_weights()

  def forward(self, x):

    block_1 = self.ReLU(self.conv2(self.ReLU(self.conv1(x))))

    block_2 = self.ReLU(self.conv4(self.ReLU(self.conv3(self.maxPool(block_1)))))

    block_3 = self.ReLU(self.conv6(self.ReLU(self.conv5(self.maxPool(block_2)))))

    block_4 = self.ReLU(self.conv8(self.ReLU(self.conv7(self.maxPool(block_3)))))
    #print(block_4.shape[2])

    block_5 = self.ReLU(self.conv10(self.ReLU(self.conv9(self.maxPool(block_4))))) # bottom-most block
    #print(block_5.shape[2])
    
    # ==== block_5 is the bottommost layer's output. Upsampling starts from here. ====  

    up_conv_1 = self.conv11(self.upsample(block_5))
    crop_1 = transforms.CenterCrop(up_conv_1.shape[2])
    block_6 = self.ReLU(self.conv13(self.ReLU(self.conv12(torch.cat((crop_1(block_4), up_conv_1), 1)))))
    #print(block_6.shape[2])

    up_conv_2 = self.conv14(self.upsample(block_6))
    crop_2 = transforms.CenterCrop(up_conv_2.shape[2])
    block_7 = self.ReLU(self.conv16(self.ReLU(self.conv15(torch.cat((crop_2(block_3), up_conv_2), 1)))))
    #print(block_7.shape[2])

    up_conv_3 = self.conv17(self.upsample(block_7))
    crop_3 = transforms.CenterCrop(up_conv_3.shape[2])
    block_8 = self.ReLU(self.conv19(self.ReLU(self.conv18(torch.cat((crop_3(block_2), up_conv_3), 1)))))
    #print(block_8.shape[2])

    up_conv_4 = self.conv20(self.upsample(block_8))
    crop_4 = transforms.CenterCrop(up_conv_4.shape[2])
    block_9 = self.conv23(self.ReLU(self.conv22(self.ReLU(self.conv21(torch.cat((crop_4(block_1), up_conv_4), 1))))))
    #print(block_9.shape[2])

    output = self.activation(block_9)

    return output
