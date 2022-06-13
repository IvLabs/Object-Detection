from init import *
from utils import *
from data import *
from model import *
from train import PATH

if __name__=="__main__":
    model = ConvNet().to(device)
    model = torch.load(PATH)
    predicts = []
    grnd_truth = []
    with torch.no_grad:
        for images, box in test_loder:
                box = box.permute(0,3,1,2)
                images, box = images.to(device), box.to(device)
                outputs = model(images)
                box2,outputs = model.trnsform(box,outputs)
                g_value, p_value = cal_nms(images,outputs,box2)
                predicts = predicts + p_value
                grnd_truth = grnd_truth + g_value
    metric = MeanAveragePrecision(class_metrics = True)
    metric.update(predicts, grnd_truth)
    evaluation = metric.compute()
