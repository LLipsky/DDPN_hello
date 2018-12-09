import argparse
import sys
from config.base_config import cfg_from_file, cfg, print_cfg, get_models_dir
import os.path as osp
import numpy as np
from utils.dictionary import Dictionary
from networks.models import Net
from train_engine import train_net
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vg network')
    parser.add_argument('--randomize', help='randomize', default=None, type=int)
    parser.add_argument('--gpu_id', help='gpu_id', default=0, type=int)
    parser.add_argument('--train_split', help='train_split', default='train', type=str)
    parser.add_argument('--val_split', help='val_split', default='val', type=str)
    parser.add_argument('--vis_pred', help='visualize prediction', default=False, type=bool)
    parser.add_argument(
            '--pretrained_model',
            help='pretrained_model',
            default= None, #osp.join(get_models_dir(''), '_iter_25000.caffemodel'),
            type=str
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        #default='config/experiments/refcoco-kld-bbox_reg.yaml',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parse_args()
    # print('Called with options:')
    # print(opts)

    # print('Using config:')
    # pprint.pprint(cfg)
    if opts.cfg_file is not None:
        cfg_from_file(opts.cfg_file)
    # cfg.IMDB_NAME = opts.imdb_name
    print_cfg()

    # train_net_path = osp.join(get_models_dir(), 'train.prototxt')
    # val_net_path = osp.join(get_models_dir(), 'val.prototxt')
    #
    train_net_path = osp.join(get_models_dir())
    val_net_path = osp.join(get_models_dir())
    opts.train_net_path = train_net_path
    opts.val_net_path = val_net_path
    #
    # if not opts.randomize:
    #     # fix the random seeds (numpy and caffe) for reproducibility
    #     np.random.seed(cfg.RNG_SEED)
    #     caffe.set_random_seed(cfg.RNG_SEED)
    # # set up caffe
    # caffe.set_mode_gpu()
    # caffe.set_device(opts.gpu_id)
    # print('initialize solver prototxt ...')
    # solver_path = get_solver_path()
    # with open(solver_path, 'w') as f:
    #     f.write(str(get_solver(opts)))
    # print('initialize train prototxt')
    qdic_dir = cfg.QUERY_DIR  # osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')
    qdic = Dictionary(qdic_dir)
    qdic.load()
    vocab_size = qdic.size()

    net = Net(opts.train_split, vocab_size, opts)
    train_model = net()
    # train_model = Net(opts.train_split, vocab_size, opts)
    with open(train_net_path, 'w') as f:
        f.write(str(train_model))

    val_model = net(opts.val_split, vocab_size, opts)
    with open(val_net_path, 'w') as f:
        f.write(str(val_model))

    train_net(solver_path, opts)
