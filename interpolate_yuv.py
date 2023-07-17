import argparse
import torch
import torchvision.transforms.functional as TF
import os
from PIL import Image
from tqdm import tqdm
import skvideo.io
from functools import partial
from utility import read_frame_yuv2rgb, tensor2rgb
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--net', type=str, default='LDMVFI')
parser.add_argument('--config', type=str, default='configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml')
parser.add_argument('--ckpt', type=str, default='ckpt.pth')
parser.add_argument('--input_yuv', type=str, default='D:\\')
parser.add_argument('--size', type=str, default='1920x1080')
parser.add_argument('--out_fps', type=int, default=60)
parser.add_argument('--out_dir', type=str, default='.')

# sampler args
parser.add_argument('--use_ddim', dest='use_ddim', default=False, action='store_true')
parser.add_argument('--ddim_eta', type=float, default=1.0)
parser.add_argument('--ddim_steps', type=int, default=200)


def main():
    args = parser.parse_args()

    # initialise model
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = model.eval()
    print('Model loaded successfully')

    # set up sampler
    if args.use_ddim:
        ddim = DDIMSampler(model)
        sample_func = partial(ddim.sample, S=args.ddim_steps, eta=args.ddim_eta, verbose=False)
    else:
        sample_func = partial(model.sample_ddpm, return_intermediates=False, verbose=False)

    # Setup output file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    _, fname = os.path.split(args.input_yuv)
    seq_name = fname.strip('.yuv')
    width, height = args.size.split('x')
    bit_depth = 16 if '16bit' in fname else 10 if '10bit' in fname else 8
    pix_fmt = '444' if '444' in fname else '420'
    try:
        width = int(width)
        height = int(height)
    except:
        print('Invalid size, should be \'<width>x<height>\'')
        return 

    outname = '{}_{}x{}_{}fps_{}.mp4'.format(seq_name, width, height, args.out_fps, args.net)
    writer = skvideo.io.FFmpegWriter(os.path.join(args.out_dir, outname), 
        inputdict={
            '-r': str(args.out_fps)
        },
        outputdict={
            '-pix_fmt': 'yuv420p',
            '-s': '{}x{}'.format(width,height),
            '-r': str(args.out_fps),
            '-vcodec': 'libx264',  #use the h.264 codec
            '-crf': '0',           #set the constant rate factor to 0, which is lossless
            '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        }
    ) 

    # Start interpolation
    print('Using model {} to upsample file {}'.format(args.net, fname))
    stream = open(args.input_yuv, 'r')
    file_size = os.path.getsize(args.input_yuv)

    # YUV reading setup
    bytes_per_frame = width*height*1.5
    if pix_fmt == '444':
        bytes_per_frame *= 2
    if bit_depth != 8:
        bytes_per_frame *= 2

    num_frames = int(file_size // bytes_per_frame)
    rawFrame0 = Image.fromarray(read_frame_yuv2rgb(stream, width, height, 0, bit_depth, pix_fmt))
    frame0 = TF.normalize(TF.to_tensor(rawFrame0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))[None,...].cuda()
    for t in tqdm(range(1, num_frames)):
        rawFrame1 = Image.fromarray(read_frame_yuv2rgb(stream, width, height, t, bit_depth, pix_fmt))
        frame1 = TF.normalize(TF.to_tensor(rawFrame1), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))[None,...].cuda()

        with torch.no_grad():
            with model.ema_scope():
                # form condition tensor and define shape of latent rep
                xc = {'prev_frame': frame0, 'next_frame': frame1}
                c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                shape = (model.channels, c.shape[2], c.shape[3])
                # run sampling and get denoised latent
                out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape)
                if isinstance(out, tuple): # using ddim
                    out = out[0]
                # reconstruct interpolated frame from latent
                out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]

        # write to output video
        writer.writeFrame(tensor2rgb(frame0)[0])
        writer.writeFrame(tensor2rgb(out)[0])

        # update frame0
        frame0 = frame1
    
    # write the last frame
    writer.writeFrame(tensor2rgb(frame1)[0])

    stream.close()
    writer.close() # close the writer


if __name__ == "__main__":
    main()
