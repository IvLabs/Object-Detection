from init import *

def filter_set(path_img,path_label):
  new_label = []
  img_list = os.listdir(path_img)
  label_list = os.listdir(path_label)
  for i in range(len(label_list)):
    pth_name = path_label +'/'+str(label_list[i])
    tree = ET.parse(pth_name)
    root = tree.getroot()
    temp_lis = root.findall('object')
    if len(temp_lis)==1:
      new_label.append(label_list[i])
  return new_label

custom_transforms = [transforms.Normalize(mean=[-0.485, -0.456,-0.406], std=[1/0.229, 1/0.224,1/0.225])]
inv_trans = torchvision.transforms.Compose(custom_transforms)

GL_NUMGRID =7
GL_CLASSES = 20
GL_NUMBBOX = 1
def make_label(bbox):
  """Convert the (x,y,w,h,cls) data of bbox into a data form (7,7,5*B+cls_num) that is convenient for calculating Loss during training
  Note that the input bbox information is in (xc,yc,w,h) format. After converting to labels, the bbox information is converted to (px,py,w,h) format"""
  gridsize = 1.0/GL_NUMGRID
  labels = np.zeros (( 7, 7, 5*GL_NUMBBOX + GL_CLASSES))   # Note that this needs to be modified according to the number of categories in different datasets
  for i in range (len(bbox)//5):
      gridx = int ( bbox [ i * 5 + 0 ] //  gridsize )   # The current bbox center falls on the gridxth grid, column
      gridy = int ( bbox [ i * 5 + 1 ] //  gridsize )   # The center of the current bbox falls on the gridyth grid, row
      # (bbox center coordinates - the coordinates of the upper left corner of the grid)/grid size ==> the relative position of the center point of the bbox
      gridpx = bbox[i*5 + 0] /gridsize - gridx
      gridpy  =  bbox [i*5 + 1] /gridsize - gridy
      # Set the grid of row gridy and column gridx to be responsible for the prediction of the current ground truth, and set the confidence and the corresponding category probability to 1
      labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 2], bbox[i * 5 + 3], 1])
      # labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 2], bbox[i * 5 + 3], 1])
      labels[gridy, gridx, 5+int(bbox[-1])] = 1
  return labels



transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2()
], bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids']))

batchsize = 1
def collate_fn(batch):
  data = [] #filled with 64 elements thorugh for loops
  target = []
  for item in batch: #batch = 64 items list one item  = [image,label]
    im = item[0]
    open_cv_image = np.array(im)
    open_cv_image = open_cv_image.copy()
    transformed = transform(image=open_cv_image, bboxes=[item[1]],category_ids = [item[1][4]])
    im = transformed['image']
    bbox = transformed['bboxes']
    label = transformed['category_ids']
    gbox = make_label(bbox[0])
    data.append(im)
    bbox = torch.tensor(gbox).squeeze()
    target.append(bbox)
  target = torch.stack(target,dim =0)
  data = torch.stack(data,dim=0)
  return [data, target]


def cal_accuracy(coords,grndth,predicted):
  y_true = []
  y_pred = []
  num_correct = torch.zeros(predicted.size()[0])
  num_total = torch.ones(predicted.size()[0])
  for i in range(predicted.size()[0]):
      pp = torch.argmax(grndth[i,5:,coords[0][i].item(),coords[1][i].item()])
      pg = torch.argmax(predicted[i,5:,coords[0][i].item(),coords[1][i].item()])
      y_true.append(pp.item())
      y_pred.append(pg.item())
      if pp==pg:
        num_correct[i] = 1
  return torch.sum(num_correct)/torch.sum(num_total),y_true,y_pred

def trnsform(output):   # a transformation function which returns normalizes the coordinates wrt to whole image 
  output = output
  box1 = torch.clone(output)
  box2 = torch.clone(output)
  tens = torch.arange(7)*output[:,4,:,:]
  box1[:,0,:,:] = (tens + output[:,0,:,:])/7 - 0.5*output[:,2,:,:]
  box1[:,1,:,:] = (tens + output[:,1,:,:])/7 - 0.5*output[:,3,:,:]
  box1[:,2,:,:] = (tens + output[:,0,:,:])/7 + 0.5*output[:,2,:,:]
  box1[:,3,:,:] = (tens + output[:,1,:,:])/7 + 0.5*output[:,3,:,:]
  box2
  return box1

def test_img(loader):
  tes_img = iter(loader)
  images,labels = tes_img.next()
  print("images",images.size())
  print("labels",labels.size())
  print(np.shape(images))
  img = images[0].squeeze()
  img = inv_trans(img)
  # img = torchvision.transforms.functional.convert_image_dtype(image= img,dtype=torch.uint8)
  img = img.numpy()
  im2display = img.transpose((1,2,0))
  plt.imshow(im2display, interpolation='nearest')

def show_results(images,outputs,box,box2,cls_list):
  
  for bs in range(5):
    img = images[bs].cpu()
    img = inv_trans(img)
    img = img.numpy()
    img = img.transpose((1,2,0))
    im2display = img.copy()
    print(outputs[bs,0,:,:])
    bux = box.cpu().numpy()
    for i in range(7):
      for j in range(7):
        if bux[bs,0,i,j]>0:
          x = i 
          y = j
          break

    b = box2.cpu()
    bx2 = b.numpy()
    out = outputs.cpu()
    out1 = out.numpy()
    print('orig',round(224*bx2[bs,0,x,y]),round(224*bx2[bs,1,x,y]),round(224*bx2[bs,2,x,y]),round(224*bx2[bs,3,x,y]))

    Pc = np.argmax(out1[bs,5:,x,y])
    pg = np.argmax(bx2[bs,5:,x,y])
    print(Pc,pg)
    bbox1 = (round(224*bx2[bs,0,x,y]),round(224*bx2[bs,1,x,y]),round(224*bx2[bs,2,x,y]),round(224*bx2[bs,3,x,y]))
    bbox = (round(224*out1[bs,0,x,y]),round(224*out1[bs,1,x,y]),round(224*out1[bs,2,x,y]),round(224*out1[bs,3,x,y]))
    print("ground",bbox1)
    print('transformed',bbox)
    cv2.rectangle(im2display, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), 1)
    cv2.rectangle(im2display, (bbox1[0],bbox1[1]), (bbox1[2],bbox1[3]), (0, 255, 0), 1)
    cv2.putText(im2display,cls_list[Pc] ,(bbox[0],bbox[1]+10), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    print('confidence-',100*out1[bs,4,x,y])
    plt.imshow(im2display)
    plt.show()