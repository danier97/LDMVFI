import random
import torch
import torchvision
import torchvision.transforms.functional as TF


def rand_crop(*args, sz):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(args[0], output_size=sz)
    out = []
    for im in args:
        out.append(TF.crop(im, i, j, h, w))
    return out


def rand_flip(*args, p):
    out = list(args)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.hflip(im)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.vflip(im)
    return out


def rand_reverse(*args, p):
    if random.random() < p:
        return args[::-1]
    else:
        return args