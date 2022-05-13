from UGATIT import UGATIT
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--isLight', type=str2bool, default=True, help='whether is full version or light version')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='the name of dataset you want use')

    parser.add_argument('--iter', type=int, default=1000, help='the max limitation of training iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch samples randomly sampled from the training set in each iterations')
    parser.add_argument('--print_period', type=int, default=1000,
                        help='number of iterations required to print an image')
    parser.add_argument('--save_period', type=int, default=100000,
                        help='number of iterations required to save a model')
    parser.add_argument('--reduce_lr', type=str2bool, default=True,
                        help='whether to reduce the learning rate halfway through the training iteration')

    parser.add_argument('--alpha', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='the weight decay parameter used in optimiser')
    parser.add_argument('--gan_loss_weight', type=int, default=1, help='gan weight of loss')
    parser.add_argument('--cycle_loss_weight', type=int, default=10, help='cycle weight in loss')
    parser.add_argument('--identity_loss_weight', type=int, default=10, help='identity weight in loss')
    parser.add_argument('--cam_loss_weight', type=int, default=1000, help='cam weight in loss')

    parser.add_argument('--channel_number', type=int, default=64,
                        help='number of channels of the first convolutional layer used in neural network')
    parser.add_argument('--resblock_num', type=int, default=4, help='resblock number')

    parser.add_argument('--img_size', type=int, default=256, help='the size of image')

    parser.add_argument('--result_path', type=str, default='results', help='file path of saving results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='device that the model is trained on (cpu or cuda)')
    parser.add_argument('--benchmark', type=str2bool, default=False, help='set benchmark on cudnn')
    parser.add_argument('--reload_model', type=str2bool, default=False,
                        help='reload model from selected result path and dataset')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_path, args.dataset, 'model'))
    check_folder(os.path.join(args.result_path, args.dataset, 'img'))
    check_folder(os.path.join(args.result_path, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
