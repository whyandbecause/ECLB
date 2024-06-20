import torch
import torch.optim as optim
import argparse
from loss_ori_LDL_3and_usc import *
#from no_cuda import *
from resnet import *
from dataset import Cifar10, Cifar100, Clothing1M
from cutout import *
import os
from torch.utils.data import DataLoader
from torch import autograd
#import torch.distributed as dist

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3, dim=1)
    softmax_targets = F.softmax(targets/3, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class LabelSmoothingCrossEntropy(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--root', default='/media/DATA3/Data/coderv/dataset/clothing1M', type=str)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--sim', default=0.4, type=float)
# parser.add_argument('--alpha', default=50, type=float)
# parser.add_argument('--beta', default=0, type=float)
# parser.add_argument('--gama', default=0.8, type=float)
parser.add_argument('--model', default="resnet18", type=str)
parser.add_argument('--save_dir', default="./checkpoints/clothing1M128_3_LSC_penalty_256000_label_loss05_I-RWNLpcian_3and_usc_sim0.6_noisechanged", type=str)
parser.add_argument('--supervision', default=True, type=bool)
#parser.add_argument('--local_rank', default=-1, type=int,
#                      help='node rank for distributed training')
args = parser.parse_args()
#dist.init_process_group(backend='nccl')
#torch.cuda.set_device(args.local_rank)
BATCH_SIZE = 32#64x64=70.74
LR = 0.002
if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir) 

#trainset = Clothing1M(root=args.root, mode="train")
testset = Clothing1M(root=args.root, mode="test")
#train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
#test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    #sampler=test_sampler
)

if args.model == "resnet18":
    model_name = resnet18
if args.model == "resnet34":
    model_name = resnet34
if args.model == "resnet50":
    model_name = resnet50
if args.model == "resnet101":
    model_name = resnet101
if args.model == "resnet152":
    model_name = resnet152


net = model_name(pretrained=True, num_classes=args.num_class)
#net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.local_rank])
net.load_state_dict(torch.load("./checkpoints/clothing1M128_cifarstyle_LSC_penalty_256000_label_loss05_I-RWNLpcian_3and_usc_sim0.6_noisechanged/resnet18.pth"))
net.to(device)
#criterion = nn.CrossEntropyLoss()
prior = torch.ones(args.num_class) / args.num_class
prior = prior.cuda()

criterion = LabelSmoothingCrossEntropy()
contra_criterion = SupConLoss()
usc_criterion = UscLoss(BATCH_SIZE)
label_criterion = LabelLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
   

    with torch.no_grad():
      correct = 0.0
      total = 0.0
      for data in testloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += float(labels.size(0))
        correct += float((predicted == labels).sum())
        acc1 = (100 * correct/total)
           
    print ("Best Accuracy", acc1)



