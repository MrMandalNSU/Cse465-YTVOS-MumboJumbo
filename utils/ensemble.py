
import torch
import torch.utils.data
from torch import nn, optim
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import functional as F
import configuration as cfg
'''

Encoder - ConvLSTM - Decoder module

'''


class MyEnsemble(nn.Module):
    def __init__(self, initializer, encoder,convlstm,decoder):
        super(MyEnsemble, self).__init__()
        self.initializer = initializer
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder
        
    def forward(self, initRGB, initMask, RGBData):
        predictedMask = []
        c0,h0 = self.initializer(torch.cat((initRGB,initMask),1))
        if cfg.shanto_debug:
            if len(RGBData.shape) == 5:
                for i in range(5):
                    rgbFrame = RGBData[:,i,:,:,:]
                    x_tilda = self.encoder(rgbFrame)
                    c_next,h_next = self.convlstm(x_tilda, h0, c0)
                    output = self.decoder(h_next)
                    c0 = c_next
                    h0 = h_next
                    predictedMask.append(output)
            else:
                rgbFrame = RGBData[:, :, :, :]
                x_tilda = self.encoder(rgbFrame)
                c_next, h_next = self.convlstm(x_tilda, h0, c0)
                output = self.decoder(h_next)
                c0 = c_next
                h0 = h_next
                predictedMask.append(output)
        else:
            for i in range(5):
                rgbFrame = RGBData[:,i,:,:,:]
                x_tilda = self.encoder(rgbFrame)
                c_next,h_next = self.convlstm(x_tilda, h0, c0)
                output = self.decoder(h_next)
                c0 = c_next
                h0 = h_next
                predictedMask.append(output)

        if cfg.cuda_enable:
            predictedMask = torch.stack(predictedMask).type(torch.FloatTensor).cuda()
        else:
            predictedMask = torch.stack(predictedMask).type(torch.FloatTensor)
        predictedMask = predictedMask.transpose(1,0)

        # added by Team-MumboJumbo
        if cfg.shanto_debug:
            if len(RGBData.shape) < 5:
                return predictedMask[0]

        return predictedMask



