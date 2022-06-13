from init import *
from utils import *

class modifydata(torch.utils.data.Dataset):
  def __init__(self,data,transform=None):
    self.data = data
    self.transforms = transform

  def __getitem__(self,index):
    self.obj = self.data[index]
    self.image = self.obj[0]
    self.image.resize((224,224))
    self.label = self.obj[1]
    bboxes = []
    width = int(self.label['annotation']['size']['width'])
    height = int(self.label['annotation']['size']['height'])
    for item in self.label['annotation']['object']:
      cls = cls_list.index(item['name'])
      bbox = []
      xmin = int(item['bndbox']['xmin'])
      xmax = int(item['bndbox']['xmax'])
      ymin = int(item['bndbox']['ymin'])
      ymax = int(item['bndbox']['ymax'])
      bbox.append(((xmin +xmax)/2)/width)
      bbox.append(((ymin +ymax)/2)/height)
      bbox.append((xmax - xmin)/width)
      bbox.append((ymax - ymin)/height)
      bbox.append(cls)
      bboxes.append(bbox)
    return self.image,bboxes

  def __len__(self):
    return len(self.data)

if __name__ =="__main__":
    pass
else:
    data = torchvision.datasets.VOCDetection(root = './data',year = '2012',image_set = 'train', download = True,transform = None)

    data2 = torchvision.datasets.VOCDetection(root = './data2',year = '2012',image_set = 'val', download = True,transform = None)


    loder = torch.utils.data.DataLoader(modifydata(data),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            collate_fn = coll_fn)
    test_loder = torch.utils.data.DataLoader(modifydata(data2),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            collate_fn = coll_fn)