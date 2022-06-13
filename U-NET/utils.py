from init import *

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]
palette = np.array(VOC_COLORMAP)

custom_transforms = [transforms.Normalize(mean=[-0.485, -0.456,-0.406], std=[1/0.229, 1/0.224,1/0.225])]
inv_trans = torchvision.transforms.Compose(custom_transforms)


transform = A.Compose([A.Resize(512,512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2()
])

def calculate_weight(loader):
  weight_map = torch.zeros(21)
  for i,(_,mask) in enumerate(loader):
    mask = mask.permute(0,3,1,2)
    index,counts = torch.unique(torch.argmax(mask,axis = 1),sorted = True,return_counts=True)
    for i in range(len(index)):
      weight_map[index[i]] = counts[i]
  weight_map = (mask.size(2)*mask.size(3)*len(loader))/(weight_map)
  return weight_map/21

def calculate_acc(grnd,predicted):
  grnd = torch.argmax(grnd,axis = 1)
  predicted = torch.argmax(predicted,axis = 1)
  x = torch.eq(grnd,predicted).int()
  acc= torch.sum(x)/(grnd.size(1)*grnd.size(1))
  return acc

def collate_fn(batch):
  data = [] #filled with 64 elements thorugh for loops
  target = []
  for item in batch: #batch = 64 items list one item  = [image,label]
    im = item[0]
    open_cv_image = np.array(im)
    open_cv_image = open_cv_image.copy()
    transformed = transform(image=open_cv_image,mask = item[1])
    im = transformed['image']
    mask = transformed['mask']
    data.append(im)
    target.append(mask)
  target = torch.stack(target,dim =0)
  data = torch.stack(data,dim=0)
  return [data, target]


def test_img(loader):
  tes_img = iter(loader)
  images,masks = tes_img.next()
  print("images",images.size())
  print("labels",masks.size())
  print(np.shape(images))
  img = images[0].squeeze()
  img = inv_trans(img)
  img = img.numpy()
  im2display = img.transpose((1,2,0))
  grnd_mask = masks.numpy().transpose[0]
  a1 = np.argmax(grnd_mask,axis = 2)
  g_mask = palette[a1]
  plt.imshow(im2display, interpolation='nearest')
  plt.imshow(g_mask)
