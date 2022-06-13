from data import train_loader
from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "./Unet_v1.pt"
# weights = calculate_weight(train_loader)

if __name__ =="__main__":
  model = Unet().to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr = 0.00005) #adjust values as per need
  criterion2= nn.CrossEntropyLoss(weight=None) # if weughts are not required then simply write None
  
  total_step = len(train_loader)
  correct_epoch = []
  loss_lis1 = []
  acc_lis = []
  num_epochs = 50

  model.train()
  for epoch in range(num_epochs):

    for i,(images,mask) in enumerate(train_loader):
      mask = mask.permute(0,3,1,2)
      images = images.to(device)
      mask = mask.to(device)
      optimizer.zero_grad() 

      #Run the forward pass
      outputs= model(images)

      loss = criterion2(outputs,mask)
      loss.backward()
      optimizer.step()

      loss_lis1.append(loss.item())
      acc = calculate_acc(mask,outputs)
      acc_lis.append(acc.item())

      if (i+1)%5 == 0:  #  to adjust the frequency of print
        print('Epoch [{}/{}],Step [{}/{}], Loss {: .4f},Accuracy: {:.2f}%'
        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),acc*100))

  torch.save(model,PATH)
  plt.plot(loss_lis1)
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.savefig('loss.png')
  plt.clf()
  plt.plot(acc_lis)
  plt.xlabel('Iterations')
  plt.ylabel('Accuracy')
  plt.savefig('accuracy.png')
  plt.clf()

  plt.plot(loss_lis1)
  plt.plot(acc_lis)
  plt.legend(['Loss','Accuracy'])
  plt.savefig('Loss_vs_Acc')
  plt.clf()

