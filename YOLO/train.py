from model import *
model = ConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.000002)
model.train()

loss_lis,acc_lis = train(model,optimizer,1)

plt.plot(loss_lis)
plt.show()

plt.plot(acc_lis)
plt.show()

PATH = "./YOLO_v1.pt"
torch.save(model,PATH)