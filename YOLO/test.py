from init import *
from data import *
from model import*


# test_img(test_loader)

model = ConvNet().to(device)
model = torch.load('./YOLO_v1.pt')
model.eval()

test_acc,ytrue,ypred,images,outputs,box= test(model)
box2 = model.trnsform(outputs)

print(metrics.precision_score(ytrue[64:],ypred[64:],average=None))
print(metrics.recall_score(ytrue[64:],ypred[64:] , average=None))

print((sum(test_acc)/len(test_acc))*100)

show_results(images,outputs,box,box2,cls_list)



