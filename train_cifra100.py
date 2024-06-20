import torch
import torch.optim as optim
import argparse
from loss_ori_LDL_3and_usc import *
from resnet import *
from dataset import Cifar10, Cifar100
from cutout import *
import os
from torch.utils.data import DataLoader
from torch import autograd

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3, dim=1)
    softmax_targets = F.softmax(targets/3, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--root', default='/media/DATA3/Data/coderv/dataset/CIFAR100/', type=str)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--noise_type', default='sym', type=str)
parser.add_argument('--noise_ratio', default=0.6, type=float)
parser.add_argument('--sim', default=0.4, type=float)
# parser.add_argument('--alpha', default=50, type=float)
# parser.add_argument('--beta', default=0, type=float)
# parser.add_argument('--gama', default=0.8, type=float)
parser.add_argument('--model', default="resnet34", type=str)
parser.add_argument('--save_dir', default="./checkpoints/cifar100_symmetric_0.6_label_loss05_I-RWNLpcian_3and_usc_sim0.4_noisechanged", type=str)
parser.add_argument('--supervision', default=True, type=bool)
args = parser.parse_args()
BATCH_SIZE = 128
LR = 0.1
if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir) 

trainset = Cifar100(root=args.root, num_class=args.num_class, mode="train", noise_type=args.noise_type, noise_ratio=args.noise_ratio)
testset = Cifar100(root=args.root, num_class=args.num_class, mode="test", noise_type='clean', noise_ratio=0)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
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


net = model_name(num_classes=args.num_class)
net.to(device)
criterion = nn.CrossEntropyLoss()
contra_criterion = SupConLoss()
usc_criterion = UscLoss(BATCH_SIZE)
label_criterion = LabelLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    epoch = 0

    y_init_path = '{}/D.npy'.format(args.save_dir)
    NUM_TRAINDATA = len(trainloader)
    if not os.path.exists(y_init_path):
        train_dataloader_temp = DataLoader(trainset, batch_size=1, shuffle=False)
        name_label_dict = {}
        for batch_idx, (_, label, file_name) in enumerate(train_dataloader_temp):
            #print(file_name[0])
            name_label_dict[file_name[0]] = F.softmax(torch.zeros(label.size(0), args.num_class).scatter_(1, label.view(-1, 1), 10), dim=1).cpu().numpy()
        np.save(y_init_path, name_label_dict)
    else:
        name_label_dict = np.load(y_init_path, allow_pickle=True).item()

    for epoch in range(args.epoch):
        if epoch in [90, 180, 280]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        net.train()
        sum_loss = 0.0
        sum_c_loss, sum_l_loss, sum_u_loss = 0, 0, 0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            with autograd.detect_anomaly():
                length = len(trainloader)
                inputs, labels, file_names = data
                bsz = labels.size(0)
                #   inputs[0] -> origin, inputs[1] -> augmented
                inputs = torch.cat([inputs[0], inputs[1]], dim=0)
                inputs, labels = inputs.cuda(), labels.cuda()
                outputss, feat_list = net(inputs)
                outputs = outputss[:bsz]
                loss = criterion(outputs, labels)
                c_loss, l_loss, u_loss = 0, 0, 0
                L = F.softmax(torch.zeros(labels.size(0), args.num_class).cuda().scatter_(1, labels.view(-1, 1), 10), dim=1)

                _, predicted = torch.max(outputss.data, 1)
                labelss = torch.cat((labels, labels), dim=0)
                labelss = labelss.contiguous().view(-1, 1)
                predicted = predicted.contiguous().view(-1, 1)
                mask = torch.eq(labelss, labelss.T).to(device)
                mask &= (torch.eq(predicted, predicted.T).to(device))
                mask &= (torch.eq(predicted, labelss.T).to(device))
                D = torch.zeros(len(file_names), args.num_class).cuda()
                for j, file_name in enumerate(file_names):
                    #print(file_name)
                    D[j] = torch.from_numpy(name_label_dict[file_name])
                D_hat = F.softmax(outputs, dim=1)
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    p1 = nn.functional.normalize(f1, dim=1)
                    z1 = nn.functional.normalize(f2, dim=1)
                    p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
                    z1 = torch.clamp(z1, 1e-4, 1.0 - 1e-4)
                    uu_loss = usc_criterion(p1, z1)
                    u_loss += uu_loss
                    representations = torch.cat([p1, z1], dim=0)
                    similarity = torch.matmul(representations, representations.t())
                    mask &= (similarity>args.sim)
                    if args.supervision:
                        cc_loss, D_hat_ = contra_criterion(features,  mask=mask, D=D, L=L)
                        c_loss += cc_loss * 5e-1
                        #print('====', D_hat_.shape)
                        l1, l2 = torch.split(D_hat_, [bsz, bsz], dim=0)
                        D_hat += (l1 + l2)/2
                        #print('----', D_hat.shape)
                        label_feature = torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], dim=1)
                        l_loss += label_criterion(label_feature, mask=mask) * 5e-1
                    else:
                        c_loss += contra_criterion(features) * 1e-1

                D_hat = D_hat * 1.0 / (len(feat_list) + 1)  # average all median features predict
                #print('+++++', D_hat.shape)
                #D_hat = (D_hat[0:BATCH_SIZE] + D_hat[BATCH_SIZE:]) / 2
                #print('-----', D_hat.shape)
                for j, file_name in enumerate(file_names):
                    name_label_dict[file_name] = D_hat[j].cpu().detach().numpy()

                loss = c_loss + l_loss + loss + u_loss
                sum_c_loss += c_loss.item()
                sum_l_loss += l_loss.item()
                sum_u_loss += u_loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += float(labels.size(0))
                correct += float(predicted.eq(labels.data).cpu().sum())
                if i % 50 == 0:
                    print('[epoch:%d, iter:%d] Loss: %.03f Constrastive Loss: %.03f Label Loss: %.03f Usc Loss: %.03f| Acc: %.4f%%'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), sum_c_loss / (i+1), sum_l_loss / (i+1), sum_u_loss / (i+1), 100 * correct / total))
        print("Waiting Test!")

        acc1 = 0
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
            if acc1 > best_acc:
                best_acc = acc1
                torch.save(net.state_dict(), args.save_dir+"/"+args.model+".pth")
        print('Test Set Accuracy: %.4f%%' % acc1, end="====")
        print ("Best Accuracy", best_acc, end="====")
        print(args.save_dir)
    print("Training Finished, TotalEPOCH=%d" % args.epoch)
    print ("Best Accuracy", best_acc)



