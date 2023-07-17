import torch
import torch.nn.functional as F
import numpy as np
import cv2
from metrics import pytorch_ssim, lpips, flolpips



def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth, pix_fmt='420'):
    if pix_fmt == '420':
        multiplier = 1
        uv_factor = 2
    elif pix_fmt == '444':
        multiplier = 2
        uv_factor = 1
    else:
        print('Pixel format {} is not supported'.format(pix_fmt))
        return

    if bit_depth == 8:
        datatype = np.uint8
        stream.seek(iFrame*1.5*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    else:
        datatype = np.uint16
        stream.seek(iFrame*3*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    if pix_fmt == '420':
        yuv = np.empty((height*3//2, width), dtype=datatype)
        yuv[0:height,:] = Y

        yuv[height:height+height//4,:] = U.reshape(-1, width)
        yuv[height+height//4:,:] = V.reshape(-1, width)

        if bit_depth != 8:
            yuv = (yuv/(2**bit_depth-1)*255).astype(np.uint8)

        #convert to rgb
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    
    else:
        yvu = np.stack([Y,V,U],axis=2)
        if bit_depth != 8:
            yvu = (yvu/(2**bit_depth-1)*255).astype(np.uint8)
        rgb = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)

    return rgb




def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)], 1)


def gaussian_kernel(sz, sigma):
    k = torch.arange(-(sz-1)/2, (sz+1)/2)
    k = torch.exp(-1.0/(2*sigma**2) * k**2)
    k = k.reshape(-1, 1) * k.reshape(1, -1)
    k = k / torch.sum(k)
    return k


def quantize(imTensor):
    return ((imTensor.clamp(-1.0, 1.0)+1.)/2.).mul(255).round()


def tensor2rgb(tensor):
    """
    Convert GPU Tensor to RGB image (numpy array)
    """
    out = []
    for b in range(tensor.shape[0]):
        out.append(np.moveaxis(quantize(tensor[b]).cpu().detach().numpy(), 0, 2).astype(np.uint8))
    return np.array(out) #(B,H,W,C)


def calc_psnr(gt, out, *args):
    """
    args:
    gt, out -- (B,3,H,W) cuda Tensors in [-1,1]
    """
    mse = torch.mean((quantize(gt) - quantize(out))**2, dim=1).mean(1).mean(1)
    return -10 * torch.log10(mse/255**2 + 1e-8) # (B,)


def calc_ssim(gt, out, *args):
    return pytorch_ssim.ssim_matlab(quantize(gt), quantize(out), size_average=False)


def calc_lpips(gt, out, *args):
    loss_fn = lpips.LPIPS(net='alex',version='0.1').cuda()
    # return loss_fn.forward(gt, out, normalize=True)
    return loss_fn.forward(quantize(gt)/255., quantize(out)/255., normalize=True)


def calc_flolpips(gt_list, out_list, inputs_list):
    '''
    gt, out - list of (B,3,H,W) cuda Tensors in [-1,1]
    inputs - list of two (B,3,H,W) cuda Tensors in [-1,1]
    e.g. gt can contain frames 1,3,5... while inputs contains frames 0,2,4,6...
    '''
    loss_fn = flolpips.FloLPIPS(net='alex',version='0.1').cuda()
    flownet = flolpips.PWCNet().cuda()
    
    scores = []
    for i in range(len(gt_list)):
        frame_ref = (gt_list[i] + 1.) / 2.
        frame_dis = (out_list[i] + 1.) / 2.
        frame_prev = (inputs_list[i] + 1.) / 2. if i == 0 else frame_next
        frame_next = (inputs_list[i+1] + 1.) / 2.
    
        with torch.no_grad():
            feat_ref = flownet.extract_pyramid_single(frame_ref)
            feat_dis = flownet.extract_pyramid_single(frame_dis)
            feat_prev = flownet.extract_pyramid_single(frame_prev) if i == 0 else feat_next
            feat_next = flownet.extract_pyramid_single(frame_next)

            # for first two frames in triplet
            flow_ref = flownet(frame_ref, frame_next, feat_ref, feat_next)
            flow_dis = flownet(frame_dis, frame_next, feat_dis, feat_next)
            flow_diff = flow_ref - flow_dis
            scores.append(loss_fn.forward(frame_ref, frame_dis, flow_diff, normalize=True).item())

            # for next two frames in triplet
            flow_ref = flownet(frame_ref, frame_prev, feat_ref, feat_prev)
            flow_dis = flownet(frame_dis, frame_prev, feat_dis, feat_prev)
            flow_diff = flow_ref - flow_dis
            scores.append(loss_fn.forward(frame_ref, frame_dis, flow_diff, normalize=True).item())

    return np.mean(scores)