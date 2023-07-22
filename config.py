import os
import yaml
import easydict
from os.path import join
from pathlib import Path

class Dataset:
    def __init__(self, path, txt_path, domains, files, source, target, prefix):
        self.path = path
        if Path(txt_path).name != 'origin_from_nas':
            txt_path = join(txt_path, f'{domains[source][0]}{domains[target][0]}')
        self.txt_path = txt_path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(txt_path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.safe_load(open(config_file))

save_config = yaml.safe_load(open(config_file))

args = easydict.EasyDict(args)

###################################################################
###################################################################
###################################################################
cuda_visible_devices = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES').split(",")))
if cuda_visible_devices == [0]:
    args.data.dataset.txt_root = f'data/{args.data.dataset.name}/true_domains'

exec_num = int(os.environ.get('exec_num'))
import itertools
if args.data.dataset.name == 'office':
    permutations = list(itertools.permutations(list(range(3)), 2))
if args.data.dataset.name == 'officehome':
    permutations = list(itertools.permutations(list(range(5)), 2))
args.data.dataset.source = permutations[exec_num][0]
args.data.dataset.target = permutations[exec_num][1]
###################################################################
###################################################################
###################################################################

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    txt_path=args.data.dataset.txt_root,
    domains=['amazon', 'dslr', 'webcam'],
    files=[
        # 'amazon.txt',
        # 'dslr.txt',
        # 'webcam.txt'
        'labeled.txt',
        'unlabeled.txt',
    ],
    source=args.data.dataset.source,
    target=args.data.dataset.target,
    prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'officehome':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    txt_path=args.data.dataset.txt_root,
    domains=['Art', 'Clipart', 'Product', 'RealWorld'],
    files=[
        # 'Art.txt',
        # 'Clipart.txt',
        # 'Product.txt',
        # 'RealWorld.txt'
        'labeled.txt',
        'unlabeled.txt',
    ],
    source=args.data.dataset.source,
    target=args.data.dataset.target,
    prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'visda2017':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    txt_path=args.data.dataset.txt_root,
    domains=['train', 'validation'],
    files=[
        'train/image_list.txt',
        'validation/image_list.txt',
    ],
    source=args.data.dataset.source,
    target=args.data.dataset.target,
    prefix=args.data.dataset.root_path)
    dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[0]
target_file = dataset.files[1]
