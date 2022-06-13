from data import test_loader
from model import *
from train import device,PATH
from utils import *

model = Unet().to(device)
model = torch.load(PATH)
test_img(test_loader)

tes_acc = []
model.eval()
with torch.no_grad():
    for i,(images,mask) in enumerate(test_loader):
        mask = mask.permute(0,3,1,2)
        images = images.to(device)
        mask = mask.to(device)
        #Run the forward pass
        outputs= model(images)
        acc = calculate_acc(mask,outputs)
        tes_acc.append(acc.item())
        grnd_mask = mask.cpu().numpy().transpose(0,2,3,1)[0]
        final_mask = outputs.cpu().numpy().transpose(0,2,3,1)[0]
        a1 = np.argmax(grnd_mask,axis = 2)
        a2 = np.argmax(final_mask,axis = 2)
        g_mask = palette[a1]
        predicted = palette[a2]
        plt.imshow(g_mask)
        plt.imshow(predicted)
        break