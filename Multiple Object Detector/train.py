from init import *
from model import *
from data import *
from utils import *

PATH = "./YOLO_v.pt"

if __name__=="__main__":
    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
    criterion = nn.MSELoss()


    total_step = len(loder)
    correct_epoch = []
    loss_lis1 = []
    acc_lis = []
    num_epochs = 10
    model.train()
    try:
        for epoch in range(num_epochs):
        # if epoch ==20:
        #   optimizer = torch.optim.Adam(model.parameters(),lr = 0.000001)
        # if epoch ==30:
        #   optimizer = torch.optim.Adam(model.parameters(),lr = 0.0000001)
            for i,(images,box) in enumerate(loder):
                box = box.permute(0,3,1,2)
                images = images.to(device)
                box = box.to(device)
                #Run the forward pass
                outputs= model(images)
                box_iou,outputs_iou  = model.trnsform(box,outputs)
                #print("labels",labels)
                # print(box[:,1,:,:])
                # with torch.autograd.detect_anomaly():
                loss = model.calculate_loss(outputs,box,box_iou,outputs_iou)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lis1.append(loss.item())
                gindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
                # pindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
                gindex = torch.stack([gindex[0],gindex[1],gindex[2]])
                # pindex = torch.stack([pindex[1],pindex[2]])
                accuracy,_,_ = cal_accuracy(gindex.to(device),box,outputs)
                acc_lis.append(accuracy)

                # print(loss.item())
                if (i+1)%2 == 0:
                    print('Epoch [{}/{}],Step [{}/{}], Loss {: .4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                        accuracy* 100))
    except KeyboardInterrupt:
        torch.save(model,PATH)

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
