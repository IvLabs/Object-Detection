from init import *

cls_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
            'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2()
], bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids']))

custom_transforms = [transforms.Normalize(mean=[-0.485, -0.456,-0.406], std=[1/0.229, 1/0.224,1/0.225])]
inv_trans = torchvision.transforms.Compose(custom_transforms)

GL_NUMGRID =7
GL_CLASSES = 20
GL_NUMBBOX = 2
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
      labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 2], bbox[i * 5 + 3], 1])
      # labels[gridy, gridx, 5+int(bbox[-1])] = 1
      labels[gridy, gridx, 10+int(bbox[i*5+4])] = 1
  return labels

batchsize = 32
def coll_fn(batch):
  data = [] #filled with 64 elements thorugh for loops
  target = []
  for item in batch: #batch = 64 items list one item  = [image,label]
    im = item[0]
    open_cv_image = np.array(im)
    open_cv_image = open_cv_image.copy()
    transformed = transform(image=open_cv_image, bboxes=item[1],category_ids = [itm[4] for itm in item[1]])
    im = transformed['image']
    bbox = transformed['bboxes']
    label = transformed['category_ids']
    # print(bbox)
    gbox = make_label([item for elem in bbox for item in elem])
    data.append(im)
    bbox = torch.tensor(gbox).squeeze()
    target.append(bbox)
  target = torch.stack(target,dim =0)
  data = torch.stack(data,dim=0)
  return [data, target]

def cal_accuracy(coords,grndth,predicted):
  y_true = []
  y_pred = []
  num_correct = []
  num_total = []
  for i in range(coords.size(1)):
      pp = torch.argmax(grndth[coords[0][i].item(),10:,coords[1][i].item(),coords[2][i].item()])
      pg = torch.argmax(predicted[coords[0][i].item(),10:,coords[1][i].item(),coords[2][i].item()])
      y_true.append(pp.item())
      y_pred.append(pg.item())
      num_total.append(1)
      if pp==pg:
        num_correct.append(1)
  return sum(num_correct)/sum(num_total),y_true,y_pred

def cal_nms(images,outputs,box):
    predicts = []
    grnd_truth = []
    torch.set_printoptions(precision=10)
    for bs in range(images.size(0)):
        box_lis = []
        grnd_box = []
        grnd_label = []
        score_lis = []
        cat_lis = []
        grnd_dict = {}
        Dict = {}
        img = images[bs].cpu()
        img = inv_trans(img)
        img = img.numpy()
        img = img.transpose((1,2,0))
        b = box.cpu()
        bx2 = b.numpy()
        out = outputs.cpu()
        out1 = out.numpy()
        im2display = img.copy()
        bux = box.cpu().numpy()
        for x in range(7):
            for y in range(7):
                if bx2[bs,4,x,y]>0:
                    grnd_box.append([round(224*bx2[bs,0,x,y]),round(224*bx2[bs,1,x,y]),round(224*bx2[bs,2,x,y]),round(224*bx2[bs,3,x,y])])
                    grnd_label.append(np.argmax(bx2[bs,10:,x,y]))
                if out1[bs,4,x,y]>0.1:
                    box_lis.append([round(224*out1[bs,0,x,y]),round(224*out1[bs,1,x,y]),round(224*out1[bs,2,x,y]),round(224*out1[bs,3,x,y])])
                    score_lis.append(out1[bs,4,x,y])
                    cat_lis.append(np.argmax(out1[bs,10:,x,y]))
                if out1[bs,9,x,y]>0.1:
                    box_lis.append([round(224*out1[bs,5,x,y]),round(224*out1[bs,6,x,y]),round(224*out1[bs,7,x,y]),round(224*out1[bs,8,x,y])])
                    score_lis.append(out1[bs,9,x,y])
                    cat_lis.append(np.argmax(out1[bs,10:,x,y]))
        filtered = torchvision.ops.batched_nms(boxes = torch.Tensor(box_lis), scores = torch.Tensor(score_lis),idxs = torch.Tensor(cat_lis), iou_threshold = 0)
        box_tens = torch.Tensor(box_lis)
        final_pred = box_tens[filtered]
        score_tens = torch.Tensor(score_lis)
        final_score= score_tens[filtered]
        cat_tens = torch.Tensor(cat_lis)
        final_cat = cat_tens[filtered]
        Dict['boxes'] = final_pred
        Dict['scores'] =  final_score
        Dict['labels'] = final_cat
        grnd_dict['boxes'] = torch.Tensor(grnd_box)
        grnd_dict['labels'] = torch.Tensor(grnd_label)
        predicts.append(Dict)
        grnd_truth.append(grnd_dict)

    return grnd_truth,predicts