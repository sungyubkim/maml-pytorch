import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='A pytorch implementation of Model Agnostic Meta-Learning (MAML)')

    # experiment settings
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--data_path', type=str, default='./../../data/mini-imagenet/', help='folder which contains image data')
    parser.add_argument('--save_path', type=str, default='./results/results_files')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=int(6e+4), help='number of outer-loops')
    parser.add_argument('--batch_size', type=int, default=4, help='number of tasks in each batch per meta-update')
    parser.add_argument('--n_way', type=int, default=5, help='number of object classes to learn')
    parser.add_argument('--k_shot', type=int, default=1, help='number of examples per class to learn from')
    parser.add_argument('--k_query', type=int, default=15, help='number of examples to evaluate on (in outer loop)')

    # training settings
    parser.add_argument('--lr_in', type=float, default=1e-2, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_out', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimizer)')
    parser.add_argument('--grad_steps_num_train', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--grad_steps_num_eval', type=int, default=10, help='number of gradient updates at test time (for evaluation)')
    parser.add_argument('--first_order', type=bool, default=True, help='use first-order approximation for MAML')

    # network settings
    parser.add_argument('--backbone', type=str, default='low-end',
    help='high-end backbone(DenseNet) or low-end(original MAML network)')
    parser.add_argument('--decoupled', type=str, default='decoupled',
    help='Decouple the base-learner and meta-learner')
    parser.add_argument('--n_channel', type=int, default=32,
    help='number of channels for each convolution operation(only low-end backbone)')
    parser.add_argument('--growth_rate', type=int, default=32,
    help='growth rate for dense block(only high-end backbone)')
    parser.add_argument('--n_block', type=int, default=3,
    help='number of dense block(only high-end backbone)')
    parser.add_argument('--block_size', type=int, default=3,
    help='number of convolution for each block(only high-end backbone)')

    # device settings
    is_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=is_cuda)
    parser.add_argument('--num_workers', type=str, default=0)


    # make argparse instance
    args = parser.parse_args()
    print('Running on device: {}'.format(args.device))
    assert args.backbone=='high-end' or args.backbone=='low-end'
    assert args.decoupled=='decoupled' or args.decoupled=='coupled'

    return args
