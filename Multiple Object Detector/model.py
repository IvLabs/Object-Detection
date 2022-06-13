from init import *
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.criterion = nn.MSELoss(reduction = 'none')
    resnet = models.resnet18(pretrained=True)
    resnet_out_channel = resnet.fc.in_features
    self.resnet = nn.Sequential(*list( resnet.children())[:-2])
    self.newlayers = nn.Sequential(nn.Linear(GL_NUMGRID * GL_NUMGRID * resnet_out_channel, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, GL_NUMGRID * GL_NUMGRID * (5*GL_NUMBBOX+GL_CLASSES)),
            nn.Sigmoid())

  def forward(self,x):
    x = self.resnet(x)
    # print('okay',x[0,:,:,2])
    x = x.view(x.size()[0], -1)
    # print('okay',x.size())
    x = self.newlayers(x)
    self.pred = x.reshape (-1,(5*GL_NUMBBOX + GL_CLASSES), GL_NUMGRID , GL_NUMGRID )   # Remember to reshape the output data at the end
    # print('okay',self.pred[0,2,:,:])
    return self.pred

  def trnsform(self,grnd,output):   # a transformation function which returns normalizes the coordinates wrt to whole image
    grnd = grnd.to(device)
    onetens = torch.ones(grnd.size()).to(device)
    output = output.to(device) 
    box1 = torch.clone(grnd).to(device)
    tensx = torch.arange(7).to(device)
    tensy = torch.arange(7).reshape(7,-1).to(device)
    tensx = tensx*grnd[:,4,:,:]
    tensy = tensy*grnd[:,4,:,:]
    for i in range(GL_NUMBBOX):
      box1[:,5*i+0,:,:] = (tensx + grnd[:,5*i+0,:,:])/7 - 0.5*grnd[:,5*i+2,:,:]
      box1[:,5*i+1,:,:] = (tensy + grnd[:,5*i+1,:,:])/7 - 0.5*grnd[:,5*i+3,:,:]
      box1[:,5*i+2,:,:] = (tensx + grnd[:,5*i+0,:,:])/7 + 0.5*grnd[:,5*i+2,:,:]
      box1[:,5*i+3,:,:] = (tensy + grnd[:,5*i+1,:,:])/7 + 0.5*grnd[:,5*i+3,:,:]

    box2 = torch.clone(output).to(device)
    tensx = torch.arange(7).to(device)
    tensy = torch.arange(7).reshape(7,-1).to(device)
    tensx = tensx*onetens[:,4,:,:]
    tensy = tensy*onetens[:,4,:,:]
    for i in range(GL_NUMBBOX):
      box2[:,5*i+0,:,:] = (tensx + output[:,5*i+0,:,:])/7 - 0.5*output[:,5*i+2,:,:]
      box2[:,5*i+1,:,:] = (tensy + output[:,5*i+1,:,:])/7 - 0.5*output[:,5*i+3,:,:]
      box2[:,5*i+2,:,:] = (tensx + output[:,5*i+0,:,:])/7 + 0.5*output[:,5*i+2,:,:]
      box2[:,5*i+3,:,:] = (tensy + output[:,5*i+1,:,:])/7 + 0.5*output[:,5*i+3,:,:]
    
    return box1,box2


  def calculate_loss(self,outputs,labels1,labels_iou,out_iou):
    self.labels = labels1.double().to(device)
    self.pred = self.pred.double()
    self.iou_labels = labels_iou.to(device)
    self.iou_pred = out_iou.to(device)
    self.coor_loss = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)     #made a empty tensor of[batch_size,7,7]
    self.obj_confi_loss = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)
    self.class_loss = torch.zeros(self.pred.size()[0], GL_CLASSES, GL_NUMGRID, GL_NUMGRID).to(device)
    self.iou1 = torch.zeros(self.pred.size()[0],GL_NUMBBOX, GL_NUMGRID, GL_NUMGRID).to(device)          #for now consider this is useless
    self.obj = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)              #made that matrix obj1 of size(batchsize,7,7)
    self.obj = self.labels[:,4,:,:].to(device)                                            #set the values of obj matrix
    # self.coor_loss = self.obj *self.criterion(self.pred[:,0,:,:],self.labels[:,0,:,:])
    self.noobj = torch.where(self.obj > 0, 0., 1.).to(device) 
    self.iou_pred = torch.reshape(self.iou_pred,(self.iou_pred.size(0),self.iou_pred.size(1),-1)).to(device) 
    self.iou_labels = torch.reshape(self.iou_labels,(self.iou_labels.size(0),self.iou_labels.size(1),-1)).to(device) 
    self.class_probs = self.pred[:,5*GL_NUMBBOX:,:,:].to(device) 
    self.box1 = self.pred[:,:5,:,:].to(device) 
    self.box2 = self.pred[:,5:10,:,:].to(device) 
    
    
    for n in range(self.pred.size(0)):
      for i in range(GL_NUMBBOX):
        self.iou1[n,i,:,:] = torch.reshape(torch.diagonal(torchvision.ops.box_iou(boxes1 = torch.t(self.iou_pred[n,(5*i+0):(5*i+4),:]), boxes2 = torch.t(self.iou_labels[n,(5*i+0):(5*i+4),:])),0),(GL_NUMGRID,-1))
    # for m in range(7):
    #   for n in range(7):
    #     self.iou1[:,m,n] = torch.diagonal(torchvision.ops.box_iou(boxes1 = self.iou_pred[:,0:4,m,n], boxes2 = self.iou_labels[:,0:4,m,n]),0)
    self.iou_indice = torch.argmax(self.iou1,dim =1).to(device) 
    self.iou_indice = torch.broadcast_to(self.iou_indice, (5,self.pred.size()[0],GL_NUMGRID,GL_NUMGRID))
    self.iou_indice = self.iou_indice.permute(1,0,2,3)
    self.box_final = torch.where(self.iou_indice>0,self.box2,self.box1).to(device) 
    self.box_min = torch.where(self.iou_indice>0,self.box1,self.box2).to(device) 
    self.pred_final = torch.cat((self.box_final,self.class_probs),dim = 1).to(device) 
    self.pred_min = torch.cat((self.box_min,self.class_probs),dim = 1).to(device) 
    self.coor_loss = 5*(self.obj*(self.pred_final[:,0,:,:] - self.labels[:,0,:,:])**2 + self.obj*(self.pred_final[:,1,:,:] - self.labels[:,1,:,:])**2 + self.obj*(torch.sqrt(self.pred_final[:,2,:,:]) - torch.sqrt(self.labels[:,2,:,:]))**2+ self.obj*(torch.sqrt(self.pred_final[:,3,:,:]) - torch.sqrt(self.labels[:,3,:,:]))**2)
    # implementation of SSE fucntion , done it in long way becoz couldn't understand how to do with nn.MSELoss()
    
    self.obj_confi_loss = self.obj*(self.pred_final[:,4,:,:] - (torch.max(self.iou1,dim = 1)[0]))**2 +0.5*self.obj*(self.pred_min[:,4,:,:]-(torch.min(self.iou1,dim = 1)[0]))**2 + 0.5*(self.noobj*(torch.sum((self.pred[:,[4,9],:,:]**2),1)))
    self.class_loss = self.obj*torch.sum((self.pred[:,5*GL_NUMBBOX:,:,:] - self.labels[:,5*GL_NUMBBOX:,:,:])**2,1)
    # + 0.5*(self.obj*(self.pred_min[:,4,:,:] - torch.min(self.iou1,dim = 1)[0])**2) 
    print(torch.sum(self.coor_loss).item(),torch.sum(self.obj_confi_loss).item(),torch.sum(self.class_loss).item())
    return (torch.sum(self.coor_loss) + torch.sum(self.obj_confi_loss) + torch.sum(self.class_loss))/self.pred.size()[0]

