import os
import torch
import torchfile
import numpy as np

from utils import *
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


DATA_PATH = ''


def imgTrainTransform(img_size):
    return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
def imgTestTransform(img_size):
    return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class CubImg_Filter(ImageFolder):
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class CubImg_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH+'/cub/cub_img', use_gpu=True, num_workers=8, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/cub/cub_img'
        
        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False

        trainset = CubImg_Filter(os.path.join(dataroot,'train'), imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = CubImg_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        outset = CubImg_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CubText_Filter(Dataset):
    def __init__(self, text_dir, text_path, class_path, traintype):
        
        def get_txt_name(m_file):
            f = open(m_file, 'r')
            content = f.read()
            f.close()
            return content.strip().split('\n')

        self.text_dir   = text_dir
        self.class_list = get_txt_name(class_path)
        self.text_names = get_txt_name(text_path)
        
        if traintype:
            nowclass = torchfile.load(os.path.join(self.text_dir, self.text_names[0]))
            num = len(nowclass)
            txts = nowclass[:int(0.8*num)]
            targets = [0] * int(0.8*num)
            for i in range(len(self.text_names)-1):
                nowclass = torchfile.load(os.path.join(self.text_dir,self.text_names[i+1]))
                num = len(nowclass)
                nowtxt = nowclass[:int(0.8*num)]
                nowtarget = [i+1] * int(0.8*num)
                txts = np.concatenate((txts,nowtxt))
                targets = np.concatenate((targets,nowtarget))
        else:
            nowclass = torchfile.load(os.path.join(self.text_dir,self.text_names[0]))
            num = len(nowclass)
            txts = nowclass[int(0.8 * num):]
            targets = [0] *(num - int(0.8 * num))
            for i in range(len(self.text_names) - 1):
                nowclass = torchfile.load(os.path.join(self.text_dir,self.text_names[i + 1]))
                num = len(nowclass)
                nowtxt = nowclass[int(0.8 * num):]
                nowtarget = [i + 1] * (num - int(0.8 * num))
                txts = np.concatenate((txts, nowtxt))
                targets = np.concatenate((targets, nowtarget))
        self.txts = txts
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        target = self.targets[i]
        txt = self.txts[i]
        return txt,target

    def __Filter__(self, known):
        datas, targets = self.txts, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if targets[i] in known:
                new_datas.append(datas[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.txts, self.targets = datas, targets


class Cubtxt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH+'/cub/cub_txt', use_gpu=True, num_workers=8, batch_size=128):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/cub/cub_txt'
        
        print('Selected Labels: ', known)
        print('OSR Labels: ', unknown)

        pin_memory = True if use_gpu else False

        trainset = CubText_Filter(os.path.join(dataroot,'word_c10'), DATA_PATH+'/manifest.txt', DATA_PATH+'/allclasses.txt' ,True)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = CubText_Filter(os.path.join(dataroot,'word_c10'),DATA_PATH+'/manifest.txt', DATA_PATH+'/allclasses.txt',False)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        outset = CubText_Filter(os.path.join(dataroot,'word_c10') ,DATA_PATH+'/manifest.txt', DATA_PATH+'/allclasses.txt',False)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CubImageTxt_Filter(ImageFolder):
    def __init__(self,root, transform):
        super().__init__(root, transform)
        self.pathdic = getDicToTXT_cub()

        def get_txt_name(m_file):
            f = open(m_file, 'r')
            content = f.read()
            f.close()
            return content.strip().split('\n')

        class_list = get_txt_name('dataprocess/cub/classes.txt')
        self.text_names = []
        for item in class_list:
            tempItem = item.split()[1]
            self.text_names.append(tempItem + '.npy')
        self.txts = allTXT(self.text_names, DATA_PATH+'/cub/cub_txt')

    def __getitem__(self, index):
        path, target = self.samples[index]
        if('train' in path):
            pathToTxt = path.split('train/')[1]
        else:
            pathToTxt = path.split('test/')[1]
        txt = self.pathdic[pathToTxt]
        sample = self.loader(path)
        sample = self.transform(sample)
        txtsample = self.txts[txt[0]][txt[1]]
        return {'img':sample,'txt':txtsample}, target
    
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class CubimageImageTxt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH+'/cub', use_gpu=True, num_workers=8, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/cub'

        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False
        
        trainset = CubImageTxt_Filter(root = os.path.join(dataroot,'cub_img','train'), transform = imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = CubImageTxt_Filter(root = os.path.join(dataroot,'cub_img','test'), transform = imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = CubImageTxt_Filter(root = os.path.join(dataroot,'cub_img','test'), transform = imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Floimage_Filter(ImageFolder):
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class Floimage_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=8, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/flower/flower_img'

        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False

        trainset = Floimage_Filter(os.path.join(dataroot,'train'), imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = Floimage_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = Floimage_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)
        
        # outset = CubImg_Filter('/root/project/cub/cub_img/test', transform)
        # outset.__Filter__(known=list(range(0, 200)))

        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Flotxt_Filter(Dataset):
    def __init__(self, text_dir,class_path, traintype):
        self.text_dir = text_dir
        def get_txt_name(m_file):
            f = open(m_file, 'r')
            content = f.read()
            f.close()
            return content.strip().split('\n')

        self.class_list = get_txt_name(class_path)
        self.text_names = []
        for item in self.class_list:
            self.text_names.append(item + '.npy')
        if traintype:
            nowclass = np.load(os.path.join(self.text_dir,self.text_names[0]))
            num = len(nowclass)
            txts = nowclass[:int(0.75*num)]
            targets = [0] * int(0.75*num)
            for i in range(len(self.text_names)-1):
                nowclass = np.load(os.path.join(self.text_dir,self.text_names[i+1]))
                num = len(nowclass)
                nowtxt = nowclass[:int(0.75*num)]
                nowtarget = [i+1] * int(0.75*num)
                txts = np.concatenate((txts,nowtxt))
                targets = np.concatenate((targets,nowtarget))
        else:
            nowclass = np.load(os.path.join(self.text_dir,self.text_names[0]))
            num = len(nowclass)
            txts = nowclass[int(0.75 * num):]
            targets = [0] *(num - int(0.75 * num))
            for i in range(len(self.text_names) - 1):
                nowclass = np.load(os.path.join(self.text_dir,self.text_names[i + 1]))
                num = len(nowclass)
                nowtxt = nowclass[int(0.75 * num):]
                nowtarget = [i + 1] * (num - int(0.75 * num))
                txts = np.concatenate((txts, nowtxt))
                targets = np.concatenate((targets, nowtarget))
        self.txts = txts
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        target = self.targets[i]
        txt = self.txts[i]
        return txt,target

    def __Filter__(self, known):
        datas, targets = self.txts, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if targets[i] in known:
                new_datas.append(datas[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.txts, self.targets = datas, targets


class Flotxt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=8, batch_size=128):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/flower/flowers_txt'
        
        print('Selected Labels: ', known)
        print('OSR Labels: ', unknown)
        pin_memory = True if use_gpu else False

        trainset = Flotxt_Filter(dataroot ,'dataprocess/flower/allclasses_flower.txt' ,True)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        testset = Flotxt_Filter(dataroot ,'dataprocess/flower/allclasses_flower.txt' ,False)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        outset = Flotxt_Filter(dataroot ,'dataprocess/flower/allclasses_flower.txt' ,False)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class FloImageTxt_Filter(ImageFolder):
    def __init__(self,root, transform):
        super().__init__(root, transform)
        self.pathdic = getDicToTXT_Flo()
        def get_txt_name(m_file):
            f = open(m_file, 'r')
            content = f.read()
            f.close()
            return content.strip().split('\n')
        class_list = get_txt_name('dataprocess/flower/allclasses_flower.txt')
        self.text_names = []
        for item in class_list:
            self.text_names.append(item + '.npy')
        self.txts = allTXT(self.text_names,DATA_PATH+'/flower/flowers_txt')
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        if('train' in path):
            pathToTxt = path.split('train/')[1]
        else:
            pathToTxt = path.split('test/')[1]
        txt = self.pathdic[pathToTxt]
        sample = self.loader(path)
        sample = self.transform(sample)
        txtsample = self.txts[txt[0]][txt[1]]
        return {'img':sample,'txt':txtsample}, target
    
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class FloImageTxt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=8, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/flower'

        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False

        trainset = FloImageTxt_Filter(root = os.path.join(dataroot,'flowers_img','train'), transform = imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = FloImageTxt_Filter(root = os.path.join(dataroot,'flowers_img','test'), transform  = imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = FloImageTxt_Filter(root = os.path.join(dataroot,'flowers_img','test'), transform = imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)

        # outset = CubImageTxt_Filter(root = DATA_PATH+'/cub/cub_img/test', transform = imgTestTransform(img_size))
        # outset.__Filter__(known=list(range(0, 200)))
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Food101image_Filter(ImageFolder):
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class Food101image_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=12, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/Food101/images'
        
        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False

        trainset = Food101image_Filter(os.path.join(dataroot,'train'), imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = Food101image_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = Food101image_Filter(os.path.join(dataroot, 'test'), imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Food101txt_Filter(Dataset):
    def __init__(self, text_path):
        self.txtPaths = []
        self.labels = []
        fileHandler = open( text_path, "r")
        while True:
            line = fileHandler.readline()
            if not line:
                break;
            nowTxtPath,nowlabel = line.strip().split()
            self.txtPaths.append(nowTxtPath)
            self.labels.append(int(nowlabel))
        fileHandler.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        target = int(self.labels[i])
        txt = np.load(self.txtPaths[i])
        return txt,target

    def __Filter__(self, known):
        paths,targets = self.txtPaths,self.labels
        newPaths, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                newPaths.append(paths[i])
                new_targets.append(known.index(targets[i]))
        self.txtPaths, self.labels = newPaths, new_targets


class Food101txt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=8, batch_size=128):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/Food101/texts_cbow'
        
        print('Selected Labels: ', known)
        print('OSR Labels: ', unknown)
        pin_memory = True if use_gpu else False

        trainset = Food101txt_Filter('dataprocess/food/train_txtpath2label.txt')
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = Food101txt_Filter('dataprocess/food/test_txtpath2label.txt')
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        outset = Food101txt_Filter('dataprocess/food/test_txtpath2label.txt')
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Food101ImageTxt_Filter(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        templist = path.split('/')
        trainOrtest = templist[-3]
        classname = templist[-2]
        imagename = templist[-1]
        txtfilename = DATA_PATH+'/Food101/texts_cbow/'+trainOrtest +'/class_'+ classname[-3:]+'/'+imagename[:-3]+'npy'
        sample = self.loader(path)
        sample = self.transform(sample)
        txtsample = np.load(txtfilename)
        
        return {'img':sample,'txt':txtsample}, target
    
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
        
class Food101ImageTxt_OSR(object):
    def __init__(self, known,unknown, dataroot=DATA_PATH, use_gpu=True, num_workers=12, batch_size=128, img_size=224):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown
        dataroot+='/Food101'

        print('Selected Labels: ', known)
        pin_memory = True if use_gpu else False

        trainset = Food101ImageTxt_Filter(root = os.path.join(dataroot,'images','train'), transform=imgTrainTransform(img_size))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        testset = Food101ImageTxt_Filter(root = os.path.join(dataroot,'images','test'), transform=imgTestTransform(img_size))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = Food101ImageTxt_Filter(root = os.path.join(dataroot,'images','test'), transform=imgTestTransform(img_size))
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))