from init import *
from utils import *
cls_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
            'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

            
data = torchvision.datasets.VOCDetection(root = './data',year = '2007',image_set = 'train', download = False,transform = None)

data2 = torchvision.datasets.VOCDetection(root = './data2',year = '2007',image_set = 'test', download = False,transform = None)

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root_img, root_label,labels):
        self.root_img = root_img
        self.root_label = root_label
        # the list of label
        self.labels = labels

    def __getitem__(self, index):
        # obtain filenames from list
        pth_name = self.root_label +'/'+str(self.labels[index])
        self.tree = ET.parse(pth_name)
        self.root = self.tree.getroot()
        image_filename = self.root[1].text
        # Load data and label
        image = Image.open(os.path.join(self.root_img, image_filename),'r')
       
        cls = cls_list.index(self.root[6][0].text)
        width = int(self.root.find('size').find('width').text)
        height = int(self.root.find('size').find('height').text)
        xmin = int(self.root.find('object').find('bndbox').find('xmin').text)
        ymin = int(self.root.find('object').find('bndbox').find('ymin').text)
        xmax = int(self.root.find('object').find('bndbox').find('xmax').text)
        ymax = int(self.root.find('object').find('bndbox').find('ymax').text)
        
        bbox = []
        bbox.append(((xmin +xmax)/2)/width)
        bbox.append(((ymin +ymax)/2)/height)
        bbox.append((xmax - xmin)/width)
        bbox.append((ymax - ymin)/height)
        bbox.append(cls)
        
        # number of objects in the image

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        # Size of bbox (Rectangular)
        # Annotation is in dictionary format
        return image,bbox

    def __len__(self):
        return len(self.labels)
 
train_data_dir = './data/VOCdevkit/VOC2007/JPEGImages'
train_label = './data/VOCdevkit/VOC2007/Annotations'
test_data_dir = './data2/VOCdevkit/VOC2007/JPEGImages'
test_label = './data2/VOCdevkit/VOC2007/Annotations'



new_label = filter_set(train_data_dir,train_label)
new_label2 = filter_set(test_data_dir,test_label)

# new_imgt , new_labelt = filter_set(test_data_dir,test_label,test_label)
# create own Dataset
train_dataset = myOwnDataset(root_img = train_data_dir,root_label = train_label ,
                          labels=new_label
                          )
test_dataset = myOwnDataset(root_img = test_data_dir,root_label = test_label ,
                          labels=new_label2
                          )


train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batchsize,
                                          shuffle=True,
                                           collate_fn = collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batchsize,
                                          shuffle=True,
                                           collate_fn = collate_fn)
# print(len(train_loader))
# print(len(test_loader))


