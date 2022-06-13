from init import *
from utils import *

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, data,transform = None):
      self.data = data
      self.transforms = transform
    @staticmethod
    def _convert_to_segmenatation_mask(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height,width,len(VOC_COLORMAP)), dtype = np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask
    def __getitem__(self, index):
        # obtain filenames from list
        # Load data and label
        self.obj = data[index]
        image = self.obj[0].convert('RGB')
        image = np.asarray(image)
        mask = self.obj[1].convert('RGB')
        mask = np.asarray(mask)
        mask = self._convert_to_segmenatation_mask(mask)
        return image,mask

    def __len__(self):
        return len(self.data)-109


# create own Datasets
batchsize = 1

if __name__ =="__main__":
    pass
else:
    data = torchvision.datasets.VOCSegmentation(root = './data',year = '2007',image_set = 'train', download = False,transform = None)

    data2 = torchvision.datasets.VOCSegmentation(root = './data2',year = '2007',image_set = 'test', download = False,transform = None)

    train_dataset = myOwnDataset(data
                            )
    test_dataset = myOwnDataset(data2
                            )


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batchsize,
                                            shuffle=True,
                                            collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batchsize,
                                            shuffle=True,
                                            collate_fn = collate_fn)