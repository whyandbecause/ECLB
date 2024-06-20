from torch.utils.data import Dataset
import os
import torch
import pandas
import numpy as np
from data_utils import noisify
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from cutout import Cutout
import random

def corrupted_labels100(targets, r = 0.4, noise_type='sym'):
    size = int(len(targets)*r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0,99))
            elif noise_type == 'asym':
                noisy_label.append((targets[i]+1)%5 + int(targets[i]/5)*5)
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x

def corrupted_labels(targets, r = 0.4, noise_type='sym'):
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                       8: 8}  # class transition for asymmetric noise
    size = int(len(targets)*r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0,9))
            elif noise_type == 'asym':
                noisy_label.append(transition[targets[i]])
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x

class Cifar100(Dataset):
    def __init__(self, root, num_class, mode="train", noise_type='sym', noise_ratio=0.1):
        super(Cifar100, self).__init__()

        # classes = {'PV': 0, 'ET': 1, 'PMF': 2, 'PrePMF': 3, 'Healthy': 4}
        self.mode = mode
        self.files, self.labels = [], []
        dirs = os.listdir(os.path.join(root, mode))
        for dir in dirs:
            files = os.listdir(os.path.join(root, mode, dir))
            for file in files:
                self.files.append(os.path.join(root, mode, dir, file))
                self.labels.append(int(dir))
        self.noise_type = noise_type
            # create noise
        if noise_type != 'clean':
            # self.noisy_labels, self.actual_noise_rate = noisify(
            #     nb_classes=num_class,
            #     train_labels=np.expand_dims(self.labels, 1),
            #     noise_type=noise_type,
            #     noise_rate=noise_ratio)
            # self.noisy_labels = self.noisy_labels.squeeze()
            self.noisy_labels = corrupted_labels100(self.labels, noise_ratio, noise_type)
            actual_noise = (self.noisy_labels != self.labels).mean()
            print("noise ratio:", actual_noise)

        self.transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.transform_origin = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx] if self.noise_type == 'clean' else self.noisy_labels[idx]
        img = Image.open(file).convert('RGB')
        if self.mode == 'train':
            img1 = self.transform_origin(img)
            img2 = self.transform_aug(img)
            return [img1, img2], label, file
        else:
            img = self.transform_val(img)
            return img, label

    def __len__(self):
        return len(self.files)


class Cifar10(Dataset):
    def __init__(self, root, num_class, mode="train", noise_type='sym', noise_ratio=0.1):
        super(Cifar10, self).__init__()

        # classes = {'PV': 0, 'ET': 1, 'PMF': 2, 'PrePMF': 3, 'Healthy': 4}
        self.mode = mode
        self.files, self.labels = [], []
        dirs = os.listdir(os.path.join(root, mode))
        for dir in dirs:
            files = os.listdir(os.path.join(root, mode, dir))
            for file in files:
                self.files.append(os.path.join(root, mode, dir, file))
                self.labels.append(int(dir))
        self.noise_type = noise_type
            # create noise
        if noise_type != 'clean':
            # self.noisy_labels, self.actual_noise_rate = noisify(
            #     nb_classes=num_class,
            #     train_labels=np.expand_dims(self.labels, 1),
            #     noise_type=noise_type,
            #     noise_rate=noise_ratio)
            # self.noisy_labels = self.noisy_labels.squeeze()
            self.noisy_labels = corrupted_labels(self.labels, noise_ratio, noise_type)
            actual_noise = (self.noisy_labels != self.labels).mean()
            print("noise ratio:", actual_noise)

        self.transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.transform_origin = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx] if self.noise_type == 'clean' else self.noisy_labels[idx]
        img = Image.open(file).convert('RGB')
        if self.mode == 'train':
            img1 = self.transform_origin(img)
            img2 = self.transform_aug(img)
            return [img1, img2], label, file, self.labels[idx]
        else:
            img = self.transform_val(img)
            return img, label

    def __len__(self):
        return len(self.files)
        
        
class Clothing1M(Dataset):
    def __init__(self, root,  mode="train"):
        super(Clothing1M, self).__init__()

        #if mode =="train":
        #    image_list = "noisy_train_key_list.txt"
        #    all_images_label = "noisy_label_kv.txt"
        #else:
        #    image_list = "clean_test_key_list.txt"
        #    all_images_label = "clean_label_kv.txt"

        self.root = root
        self.mode = mode
        #with open(os.path.join(root, image_list)) as image_f:
        #    self.images = [l.strip() for l in image_f]

        #self.all_imgs = {}
        #with open(os.path.join(root, all_images_label)) as img_label_f:
        #    for l in img_label_f:
        #        img_label = l.strip().split()
        #        self.all_imgs[img_label[0]] = int(img_label[1])
        
#---------------------        
        self.labels = {}
        imgs = []
        self.subimgs = []
        if mode == "train":
          with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0]
                self.labels[img_path] = int(entry[1])
                
          with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                img_path = '%s/' % self.root + l
                imgs.append(img_path)
                
          random.shuffle(imgs)
          class_num = torch.zeros(14)
          for impath in imgs:
            label = self.labels[impath]
            if class_num[label] < (256000 / 14) and len(self.subimgs) < 256000:
                self.subimgs.append(impath)
                class_num[label] += 1
          random.shuffle(self.subimgs)
        else:
          with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
              entry = l.split()
              img_path = '%s/' % self.root + entry[0]
              self.labels[img_path] = int(entry[1])
              
          with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
              img_path = '%s/' % self.root + l
              self.subimgs.append(img_path)
        print(len(self.subimgs))
#-----------------------        
        self.transform_weak = transforms.Compose(
            [
                transforms.Resize((128,128)),
                #transforms.RandomCrop(224),
                transforms.RandomCrop(128, padding=8, fill=128),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        self.transform_strong = transforms.Compose(
            [
                transforms.Resize((128,128)),
                #transforms.RandomCrop(224),
                transforms.RandomCrop(128, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        self.transform_val = transforms.Compose([
            transforms.Resize((128,128)),
            #transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

    def __getitem__(self, idx):
        #file = os.path.join(self.root, self.images[idx])
        #label = self.all_imgs[self.images[idx]]
        file = self.subimgs[idx]
        label = self.labels[file]
        img = Image.open(file).convert('RGB')
        if self.mode == 'train':
            img1 = self.transform_weak(img)
            img2 = self.transform_strong(img)
            return [img1, img2], label, file
        else:
            img = self.transform_val(img)
            return img, label

    def __len__(self):
        return len(self.subimgs)
        
        
class Animal10(Dataset):
    def __init__(self, root,  mode="train"):
        super(Animal10, self).__init__()

        self.root = root
        self.mode = mode
        self.images = os.listdir(os.path.join(self.root, self.mode))

        self.transform_weak = transforms.Compose(
            [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.transform_strong = transforms.Compose(
            [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        

    def __getitem__(self, idx):
        img_p = os.path.join(self.root, self.mode, self.images[idx])
        label = int(self.images[idx].split('_')[0])
        img = Image.open(img_p).convert('RGB')
        if self.mode == 'train':
            img1 = self.transform_weak(img)
            img2 = self.transform_strong(img)
            return [img1, img2], label, img_p
        else:
            img = self.transform_val(img)
            return img, label

    def __len__(self):
        return len(self.images)
