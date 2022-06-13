from init import *
from utils import *
from data import *
from model import *
from train import PATH
if __name__=="__main__":
    model = ConvNet().to(device)
    model = torch.load(PATH)
    tes_acc = []
    time_list = [] 
    model.eval()
    ytrue = np.arange(0,64)
    ypred = np.arange(0,64)
    with torch.no_grad():
        for images, box in test_loder:
            box = box.permute(0,3,1,2)
            images, box = images.to(device), box.to(device)
            begin = time.time()
            outputs = model(images)
            end = time.time()
            time_list.append(end-begin)
            box2,outputs = model.trnsform(box,outputs)
            gindex = (box2[:,0,:,:]>0).nonzero(as_tuple=True)
            # pindex = (box[:,0,:,:]>0).nonzero(as_tuple=True)
            g_index = torch.stack([gindex[0],gindex[1],gindex[2]])
            # pindex = torch.stack([pindex[1],pindex[2]])
            accuracy,grd,pred = cal_accuracy(g_index.to(device),box2,outputs)
            ytrue = np.concatenate((ytrue,np.array(grd)),axis = 0)
            ypred = np.concatenate((ypred,np.array(pred)),axis = 0)
            tes_acc.append(accuracy)
    print((sum(tes_acc)/len(tes_acc))*100)
    print((sum(time_list)/len(time_list)))
    