from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torch
from matching import compare_two_videos
import warnings
import argparse

warnings.filterwarnings("ignore")

# парсим переменные
parser = argparse.ArgumentParser("video_matcher")
parser.add_argument("-s0", "--source0", help="source of first video", type=str, required=True)
parser.add_argument("-s1", "--source1", help="source of second video", type=str, required=True)
parser.add_argument("-sl", "--simlowthres", help="lower threshold of similarity", default=0.99, type=float, required=False)
parser.add_argument("-sh", "--simhighthres", help="higher threshold of similarity", default=0.999, type=float, required=False)
parser.add_argument("-wl", "--wdlowthres", help="lower threshold of weight distance", default=10, type=float, required=False)
parser.add_argument("-wh", "--wdhighthres", help="higher threshold of weight distance", default=30, type=float, required=False)
parser.add_argument("-v", "--showvideo", help="show result video", type=bool, default=False, required=False)
args = parser.parse_args()

source0 = args.source0
source1 = args.source1
sim_low_thres = args.simlowthres
sim_high_thres = args.simhighthres
wd_low_thres = args.wdlowthres
wd_high_thres = args.wdhighthres
show_video = args.showvideo

# Используем модель предложенную в задании, так как она хорошо работает
model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

print(f'Start matching {source0} and {source1}')
print('cuda is available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    model.to('cuda')

compare_two_videos(model, source0, source1, sim_low_thres, sim_high_thres, wd_low_thres, wd_high_thres, show_video)

print(f'Matching of {source0} and {source1} finished!')