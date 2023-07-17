import argparse
import os
from ldm.data import testsets_vqm


parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--exp', type=str, default=None)
parser.add_argument('--dataset', type=str, default='Middlebury_others')
parser.add_argument('--metrics', nargs='+', type=str, default=['FloLPIPS'])
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--resume', dest='resume', default=False, action='store_true')


def main():

    args = parser.parse_args()
    
    # initialise model
    model = args.exp
    print('Evaluating model:', model)

    # setup output dirs
    assert os.path.exists(args.out_dir), 'Frames not previously interpolated!'
    
    # initialise test set
    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)
    assert os.path.exists(test_dir), f'{args.dataset} not pre-computed!'

    if args.dataset.split('_')[0] in ['VFITex', 'Ucf101', 'Davis90']:
        db_folder = args.dataset.split('_')[0].lower()
    else:
        db_folder = args.dataset.lower()

    test_db = getattr(testsets_vqm, args.dataset)(os.path.join(args.data_dir, db_folder))
    test_db.eval(metrics=args.metrics, output_dir=test_dir, resume=args.resume)



if __name__ == '__main__':
    main()