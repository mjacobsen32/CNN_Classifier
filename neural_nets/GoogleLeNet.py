import torch.nn as nn
import torch

class Inception(nn.Module):
    def __init__(self,in_ch,out_ch1,mid_ch13,out_ch13,mid_ch15,out_ch15,out_ch_pool_conv,auxiliary=False):
        super(Inception,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch1,kernel_size=1,stride=1),
            nn.ReLU())
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch13,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch13,out_ch13,kernel_size=3,stride=1,padding=1),
            nn.ReLU())
        
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch15,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch15,out_ch15,kernel_size=5,stride=1,padding=2),
            nn.ReLU())
        
        self.pool_conv1 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch_pool_conv,kernel_size=1,stride=1),
            nn.ReLU())
        
        self.auxiliary = auxiliary
        
        if auxiliary:
            self.auxiliary_layer = nn.Sequential(
                nn.AvgPool2d(5,3),
                nn.Conv2d(in_ch,128,1),
                nn.ReLU())
        
    def forward(self,inputs,train=False):
        conv1_out = self.conv1(inputs)
        conv13_out = self.conv13(inputs)
        conv15_out = self.conv15(inputs)
        pool_conv_out = self.pool_conv1(inputs)
        outputs = torch.cat([conv1_out,conv13_out,conv15_out,pool_conv_out],1) # depth-wise concat
        
        if self.auxiliary:
            if train:
                outputs2 = self.auxiliary_layer(inputs)
            else:
                outputs2 = None
            return outputs, outputs2
        else:
            return outputs


class GoogLeNet(nn.Module):
    
    def __init__(self,num_output):
        super(GoogLeNet,self).__init__()
        self.stem_layer = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(64,64,1),
            nn.ReLU(),
            nn.Conv2d(64,192,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )
        
        #in_ch,out_ch_1,mid_ch_13,out_ch_13,mid_ch_15,out_ch_15,out_ch_pool_conv
        self.inception_layer1 = nn.Sequential(
            Inception(192,64,96,128,16,32,32),
            Inception(256,128,128,192,32,96,64),
            nn.MaxPool2d(3,2,1)
        )
        
        self.inception_layer2 = nn.Sequential(
            Inception(480,192,96,208,16,48,64),
            Inception(512,160,112,224,24,64,64),
            Inception(512,128,128,256,24,64,64),
            Inception(512,112,144,288,32,64,64),
            Inception(528,256,160,320,32,128,128),
            nn.MaxPool2d(3,2,1)
        )
        
        #self.inception_layer3 = Inception(528,256,160,320,32,128,128,True) # auxiliary classifier
        #self.auxiliary_layer = nn.Linear(128*4*4,num_output)
        
        self.inception_layer3 = nn.Sequential(
            #nn.MaxPool2d(3,2,1),
            Inception(832,256,160,320,32,128,128),
            Inception(832,384,192,384,48,128,128),
            nn.AvgPool2d(7,1)
        )
        
        self.dropout = nn.Dropout2d(0.4)
        self.output_layer = nn.Linear(1024,num_output)
        
    def forward(self,inputs,train=False):
        outputs = self.stem_layer(inputs)
        outputs = self.inception_layer1(outputs)
        outputs = self.inception_layer2(outputs)
        #outputs,outputs2 = self.inception_layer3(outputs)
        #if train:
            # B,128,4,4 => B,128*4*4
        #    outputs2 = self.auxiliary_layer(outputs2.view(inputs.size(0),-1))
        outputs = self.inception_layer3(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.view(outputs.size(0),-1) # 동일 : outputs = outputs.view(batch_size,-1)
        outputs = self.output_layer(outputs)
        
        #if train:
        #   return outputs, outputs2
        return outputs