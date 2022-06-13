from init import *

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def auto_crop(self,x1,x2):
    x1dim = x1.shape[2:]
    x2dim = x2.shape[2:]
    cropped_out = x1[:,:,((x1dim[0]-x2dim[0])//2):((x1dim[0]+x2dim[0])//2),((x1dim[1]-x2dim[1])//2):((x1dim[1]+x2dim[1])//2)]
    return cropped_out

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(64),nn.ReLU()
            
        )
        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(128),nn.ReLU()
        )
        self.layer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(256),nn.ReLU()
        )

        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(512),nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(512),nn.ReLU(),
        )

        self.layer5 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(1024),nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(1024),nn.ReLU(),
            nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(512),nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(512),nn.ReLU(),
            nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(256),nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(128),nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias = False),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,21,kernel_size=1,stride=1,padding=0)
        )
    @staticmethod
    def auto_crop(x1,x2):
        x1dim = x1.shape[2:]
        x2dim = x2.shape[2:]
        cropped_out = x1[:,:,((x1dim[0]-x2dim[0])//2):((x1dim[0]+x2dim[0])//2),((x1dim[1]-x2dim[1])//2):((x1dim[1]+x2dim[1])//2)]
        return cropped_out


    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        cropped4 = self.auto_crop(out4,out5)
        out6 = self.layer6(torch.cat((cropped4,out5), 1))
        cropped3 = self.auto_crop(out3,out6)
        out7 = self.layer7(torch.cat((cropped3,out6), 1))
        cropped2 = self.auto_crop(out2,out7)
        out8 = self.layer8(torch.cat((cropped2,out7), 1))
        cropped1 = self.auto_crop(out1,out8)
        out9 = self.final_layer(torch.cat((cropped1,out8), 1))
        return out9