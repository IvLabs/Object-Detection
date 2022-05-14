from init import *
from data import *


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

  def trnsform(self,output):   # a transformation function which returns normalizes the coordinates wrt to whole image
    output = output.to(device) 
    box = torch.clone(output).to(device)
    tensx = torch.arange(7).to(device)
    tensy = torch.arange(7).reshape(7,-1).to(device)
    tensx = tensx*output[:,4,:,:]
    tensy = tensy*output[:,4,:,:]
    box[:,0,:,:] = (tensx + output[:,0,:,:])/7 - 0.5*output[:,2,:,:]
    box[:,1,:,:] = (tensy + output[:,1,:,:])/7 - 0.5*output[:,3,:,:]
    box[:,2,:,:] = (tensx + output[:,0,:,:])/7 + 0.5*output[:,2,:,:]
    box[:,3,:,:] = (tensy + output[:,1,:,:])/7 + 0.5*output[:,3,:,:]
    return box


  def calculate_loss(self,outputs,labels1,labels_iou):
    self.labels = labels1.double().to(device)
    self.pred = self.pred.double()
    self.iou_labels = labels_iou.to(device)
    self.iou_pred = outputs.to(device)
    self.coor_loss = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)     #made a empty tensor of[batch_size,7,7]
    self.obj_confi_loss = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)
    self.class_loss = torch.zeros(self.pred.size()[0], GL_CLASSES, GL_NUMGRID, GL_NUMGRID).to(device)
    self.iou1 = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)          #for now consider this is useless
    
    self.obj = torch.zeros(self.pred.size()[0], GL_NUMGRID, GL_NUMGRID).to(device)              #made that matrix obj1 of size(batchsize,7,7)
    self.obj = self.labels[:,4,:,:]                                           #set the values of obj matrix
    # self.coor_loss = self.obj *self.criterion(self.pred[:,0,:,:],self.labels[:,0,:,:])
    self.noobj = self.labels[:,4,:,:] 
    for m in range(7):
      for n in range(7):
        self.iou1[:,m,n] = torch.diagonal(torchvision.ops.box_iou(boxes1 = self.iou_pred[:,0:4,m,n], boxes2 = self.iou_labels[:,0:4,m,n]),0)
        
    self.coor_loss = 5*(self.obj*(self.pred[:,0,:,:] - self.labels[:,0,:,:])**2 + self.obj*(self.pred[:,1,:,:] - self.labels[:,1,:,:])**2 + self.obj*(torch.sqrt(self.pred[:,3,:,:]) - torch.sqrt(self.labels[:,3,:,:]))**2+ self.obj*(torch.sqrt(self.pred[:,4,:,:]) - torch.sqrt(self.labels[:,4,:,:]))**2)
    # implementation of SSE fucntion , done it in long way becoz couldn't understand how to do with nn.MSELoss()
    
    self.obj_confi_loss = self.obj*(self.pred[:,4,:,:] - self.iou1)**2 + 0.5*(self.noobj*(self.pred[:,4,:,:] - self.iou1)**2)
    self.class_loss = self.obj*torch.sum((self.pred[:,5:,:,:] - self.labels[:,5:,:,:])**2,1)
    # print(self.coor_loss[0])
    return (torch.sum(self.coor_loss) + torch.sum(self.obj_confi_loss) + torch.sum(self.class_loss))/self.pred.size()[0]

def train(model,optimizer,num_epochs):
  total_step = len(train_loader)
  correct_epoch = []
  loss_lis1 = []
  acc_lis = []
  for epoch in range(num_epochs):

    for i,(images,box) in enumerate(train_loader):
      box = box.permute(0,3,1,2)
      images = images.to(device)
      box = box.to(device)
      box_iou = model.trnsform(box).to(device)
      #Run the forward pass
      outputs= model(images)
      
      outputs = model.trnsform(outputs)
      
      #print("labels",labels)
      # print(box[:,1,:,:])
      # with torch.autograd.detect_anomaly():
      loss = model.calculate_loss(outputs,box,box_iou)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_lis1.append(loss.item())
      gindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
      # pindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
      gindex = torch.stack([gindex[1],gindex[2]])
      # pindex = torch.stack([pindex[1],pindex[2]])
      accuracy,_,_ = cal_accuracy(gindex.to(device),box,outputs)
      acc_lis.append(accuracy.item())
      

      # print(loss.item())
      if (i+1)%30 == 0:
        print('Epoch [{}/{}],Step [{}/{}], Loss {: .4f}, Accuracy: {:.2f}%'
        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            accuracy* 100))
  
  return loss_lis1,acc_lis

def test(model):
  tes_acc = []
  ytrue = np.arange(0,64)
  ypred = np.arange(0,64)
  with torch.no_grad():
      for images, box in test_loader:
        box = box.permute(0,3,1,2)
        images, box = images.to(device), box.to(device)
        begin = time.time()
        outputs = model(images)
        end = time.time()
        print("time-",end-begin)
        outputs = model.trnsform(outputs)
        box2 = model.trnsform(box)
        gindex = (box2[:,0,:,:]>0).nonzero(as_tuple=True)
        # pindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
        g_index = torch.stack([gindex[1],gindex[2]])
        # pindex = torch.stack([pindex[1],pindex[2]])
        accuracy,grd,pred = cal_accuracy(g_index.to(device),box2,outputs)
        ytrue = np.concatenate((ytrue,np.array(grd)),axis = 0)
        ypred = np.concatenate((ypred,np.array(pred)),axis = 0)
        tes_acc.append(accuracy.item())

  return tes_acc,ytrue,ypred,images,outputs,box
