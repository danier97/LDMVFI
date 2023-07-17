import numpy as np
import random
from os import listdir
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import ldm.data.vfitransforms as vt
from functools import partial

class Vimeo90k_triplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True):
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'sep_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index],  "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index],  "im5.png"))

        if self.crop_sz is not None:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)

        if self.augment_s:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5, p=0.5)
        
        if self.augment_t:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,256,3), 0-255

        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}

    def __len__(self):
        return len(self.seq_path_list)


class Vimeo90k_quintuplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True):
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'sep_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index],  "im1.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index],  "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index],  "im5.png"))
        rawFrame7 = Image.open(join(self.seq_path_list[index],  "im7.png"))

        if self.crop_sz is not None:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_crop(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, sz=self.crop_sz)

        if self.augment_s:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_flip(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)
        
        if self.augment_t:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_reverse(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)

        frame1, frame3, frame4, frame5, frame7 = map(TF.to_tensor, (rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7))

        return frame1, frame3, frame4, frame5, frame7

    def __len__(self):
        return len(self.seq_path_list)

    
class BVIDVC_triplet(Dataset):
    def __init__(self, db_dir, res=None, crop_sz=(256,256), augment_s=True, augment_t=True):

        db_dir = join(db_dir, 'quintuplets')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir)]

    def __getitem__(self, index):

        cat = Image.open(join(self.seq_path_list[index], 'quintuplet.png'))

        rawFrame3 = cat.crop((256, 0, 256*2, 256))
        rawFrame5 = cat.crop((256*2, 0, 256*3, 256))
        rawFrame4 = cat.crop((256*4, 0, 256*5, 256))

        if self.crop_sz is not None:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)

        if self.augment_s:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5, p=0.5)
        
        if self.augment_t:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,256,3), 0-255

        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}

    def __len__(self):
        return len(self.seq_path_list)


class BVIDVC_quintuplet(Dataset):
    def __init__(self, db_dir, res=None, crop_sz=(256,256), augment_s=True, augment_t=True):

        db_dir = join(db_dir, 'quintuplets')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir)]

    def __getitem__(self, index):

        cat = Image.open(join(self.seq_path_list[index], 'quintuplet.png'))

        rawFrame1 = cat.crop((0, 0, 256, 256))
        rawFrame3 = cat.crop((256, 0, 256*2, 256))
        rawFrame5 = cat.crop((256*2, 0, 256*3, 256))
        rawFrame7 = cat.crop((256*3, 0, 256*4, 256))
        rawFrame4 = cat.crop((256*4, 0, 256*5, 256))

        if self.augment_s:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_flip(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)
        
        if self.augment_t:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_reverse(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)

        frame1, frame3, frame4, frame5, frame7 = map(TF.to_tensor, (rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7))

        return frame1, frame3, frame4, frame5, frame7

    def __len__(self):
        return len(self.seq_path_list)


class Sampler(Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch


class BVI_Vimeo_triplet(Dataset):
    def __init__(self, db_dir, crop_sz=[256,256], p_datasets=None, iter=False, samples_per_epoch=1000):
        vimeo90k_train = Vimeo90k_triplet(join(db_dir, 'vimeo_septuplet'), train=True,  crop_sz=crop_sz)
        bvidvc_train = BVIDVC_triplet(join(db_dir, 'bvidvc'), crop_sz=crop_sz)

        self.datasets = [vimeo90k_train, bvidvc_train]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch