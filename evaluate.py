import argparse
import os
import torch
from functools import partial
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data import testsets


parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--config', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dataset', type=str, default='Middlebury_others')
parser.add_argument('--metrics', nargs='+', type=str, default=['PSNR', 'SSIM', 'LPIPS'])
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--resume', dest='resume', default=False, action='store_true')

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

    # setup output dirs
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # initialise test set
    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)
    if args.dataset.split('_')[0] in ['VFITex', 'Ucf101', 'Davis90']:
        db_folder = args.dataset.split('_')[0].lower()
    else:
        db_folder = args.dataset.lower()
    test_db = getattr(testsets, args.dataset)(os.path.join(args.data_dir, db_folder))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_db.eval(model, sample_func, metrics=args.metrics, output_dir=test_dir, resume=args.resume)



if __name__ == '__main__':
    main()