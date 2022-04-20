import torch
import torch.nn as nn
import sak
import sak.torch
import sak.torch.nn

""" 
Convolutional block:
It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
This is what happens in each level of the U-net!
"""
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()
        self.n_conv = n_conv
        
        setattr(self, 'conv0', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) # the first conv is different (in_channels)
        
        for i in range(n_conv):
            setattr(self, 'relu'+str(i), nn.LeakyReLU())
            setattr(self, 'bn'+str(i), nn.BatchNorm2d(out_channels))
            setattr(self, 'dropout'+str(i), nn.Dropout2d(0.25))
            if i > 0:
                setattr(self, 'conv'+str(i), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = getattr(self,'conv0')(x)
        
        for i in range(self.n_conv):
            if i > 0:
                x = getattr(self, 'conv'+str(i))(x)
            x = getattr(self, 'relu'+str(i))(x)
            x = getattr(self, 'bn'+str(i))(x)
            x = getattr(self, 'dropout'+str(i))(x)

        return x

""" 
Encoder block:
It consists of an conv_block followed by a max pooling.
Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()
        
        self.conv = conv_block(in_channels, out_channels, n_conv)
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        
        # here x acts also as a skip connection
        # p goes down a level of the network
        x = self.conv(x)
        p = self.pool(x)

        # the h and w of p is half of x!
        return x, p

""" 
Decoder block:
The decoder block begins with a transpose convolution 2x2, followed by a concatenation with the skip
connection from the encoder block. Next comes the conv_block.
Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()

        # stride allows to double the h and w
        # no need of padding as we are obtaining bigger feature matrix
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        # the input channels are double because we have the ones coming from the up-conv and the ones coming from the skip
        self.conv = conv_block(out_channels*2, out_channels, n_conv)

    def forward(self, x, skip):
        # the decoder block receives as inputs the up-conv and the skip connection!
        x = self.up(x) # up-conv
        x = torch.cat([x, skip], axis=1) # concatenation of the skip connection, axis = 1 because the concatenation is done in the number of channels
        x = self.conv(x)

        return x


class basenet(nn.Module):
    def __init__(self, n_channels, i_channels, n_levels, n_conv):
        super().__init__()

        self.n_levels = n_levels
        
        """ Encoder """
        
        # list with the operations of the encoder
        operations_encoder = [encoder_block(i_channels,n_channels, n_conv)]
        
        for i in range(1,n_levels-1):
            operations_encoder.append(encoder_block(n_channels*2**(i-1), n_channels * 2**i, n_conv))
        for i in range(len(operations_encoder)):
            setattr(self, 'e'+str(i),operations_encoder[i])
        
        """ Bottleneck: bottom part of the diagram """
        # just another convolution block
        self.b = conv_block(n_channels*2**(n_levels-2), n_channels*2**(n_levels-1), n_conv)

        """ Decoder """
        
        # list with the operations of the encoder
        operations_decoder = [decoder_block(n_channels*2**i, n_channels * 2**(i-1), n_conv) for i in range(n_levels-1,0,-1)]
        for i in range(len(operations_decoder)):
            setattr(self, 'd'+str(i),operations_decoder[i])

    def forward(self, x):
        """ Encoder """
        
        conv0_result = self.e0(x)
        p_outputs = [conv0_result[1]]
        s_outputs = [conv0_result[0]]
        
        for i in range(1,self.n_levels-1):
            conv_op = getattr(self,'e'+str(i)) 
            conv_result = conv_op(p_outputs[i-1]) 
            p_outputs.append(conv_result[1]) 
            s_outputs.append(conv_result[0]) 

        """ Bottleneck """
        b = self.b(p_outputs[self.n_levels-2])

        """ Decoder """
        d_outputs = [self.d0(b, s_outputs[self.n_levels-2])]
        for i in range(1,self.n_levels-1):
            d_outputs.append(getattr(self,'d'+str(i))(d_outputs[i-1], s_outputs[-(i+1)]))

        return d_outputs[self.n_levels-2]

class unet(nn.Module):
    def __init__(self, n_channels, i_channels, f_channels, n_levels, n_conv):
        super().__init__()
        
        self.basenet = basenet(n_channels, i_channels, n_levels, n_conv)
        
        """ Classifier: last convolution 1x1 """
        self.outputs = nn.Conv2d(n_channels, f_channels, kernel_size=1, padding=0)
        
        """ Sigmoid """
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.basenet(x)
        
        """ Classifier """
        outputs = self.outputs(x)
        outputs = self.sigmoid(outputs)
        
        return outputs

class unet_lr(nn.Module):
    def __init__(self, n_channels, i_channels, f_channels, n_levels, n_conv):
        super().__init__()
        
        self.basenet = basenet(n_channels, i_channels, n_levels, n_conv)
        
        """ Classifier: last convolution 1x1 """
        self.outputs = nn.Conv2d(n_channels, f_channels, kernel_size=1, padding=0)
        
        """ Sigmoid """
        self.sigmoid = nn.Sigmoid()

        ''' Linear regression '''
        self.lr = sak.torch.nn.SoftArgmaxAlongAxis(2)
        
    def forward(self, x):
        
        x = self.basenet(x)
        
        """ Classifier """
        outputs = self.outputs(x)
        outputs = self.sigmoid(outputs)
        outputs = self.lr(outputs) # linear regresion
        
        return outputs

class wnet(nn.Module):
    def __init__(self, n_channels, i_channels, f_channels, n_levels, n_conv):
        super().__init__()
        
        """ Encoder """
        self.encoder = basenet(n_channels, i_channels, n_levels, n_conv)
        
        """ Middle:  softmax activation function """
        self.softmax = nn.Softmax()
        
        """ Decoder """
        self.decoder = basenet(n_channels, n_channels, n_levels, n_conv)
        
        """ Classifier: last convolution 1x1 """
        self.outputs = nn.Conv2d(n_channels, f_channels, kernel_size=1, padding=0)
        
        """ Sigmoid """
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.softmax(x)
        x = self.decoder(x)
        
        """ Classifier """
        outputs = self.outputs(x)
        outputs = self.sigmoid(outputs)
        
        return outputs

class wnet_lr(nn.Module):
    def __init__(self, n_channels, i_channels, f_channels, n_levels, n_conv):
        super().__init__()
        
        """ Encoder """
        self.encoder = basenet(n_channels, i_channels, n_levels, n_conv)
        
        """ Middle:  softmax activation function """
        self.softmax = nn.Softmax()
        
        """ Decoder """
        self.decoder = basenet(n_channels, n_channels, n_levels, n_conv)
        
        """ Classifier: last convolution 1x1 """
        self.outputs = nn.Conv2d(n_channels, f_channels, kernel_size=1, padding=0)
        
        """ Sigmoid """
        self.sigmoid = nn.Sigmoid()

        ''' Linear regression '''
        self.lr = sak.torch.nn.SoftArgmaxAlongAxis(2)
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.softmax(x)
        x = self.decoder(x)
        
        """ Classifier """
        outputs = self.outputs(x)
        outputs = self.sigmoid(outputs)
        outputs = self.lr(outputs) # linear regresion
        
        return outputs