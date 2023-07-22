from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler


class DomainFileListDataset(FileListDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id domain_id
    image_path label_id domain_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        super().__init__(list_path, path_prefix, transform, return_id, num_classes, filter)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        file = ans[0].strip()
                        label = ans[1]
                        domain = ans[2]
                        data.append([file, label, domain])
            self.datas = [join_path(self.path_prefix, 'imgs', x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
                self.domains = [int(x[2]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y, d) for (x, y, d) in zip(self.datas, self.labels, self.domains) if filter(y)]
        self.datas, self.labels, self.domains = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1
    
    def __getitem__(self, index):
        im = Image.open(self.datas[index]).convert('RGB')
        im = self.transform(im)
        if not self.return_id:
            return im, self.labels[index], self.domains[index]
        return im, self.labels[index], self.domains[index], index

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
# a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
# c = c - a - b
# common_classes = [i for i in range(a)]
# source_private_classes = [i + a for i in range(b)]
# target_private_classes = [i + a + b for i in range(c)]

# source_classes = common_classes + source_private_classes
# target_classes = common_classes + target_private_classes


train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

source_train_ds = DomainFileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform)
                            # transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = DomainFileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=test_transform)
                            # transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = DomainFileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform)
                            # transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = DomainFileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform)
                            # transform=test_transform, filter=(lambda x: x in target_classes))

source_classes = sorted(np.unique(source_train_ds.labels))
target_classes = sorted(np.unique(target_train_ds.labels))
print(f'''
        source_classes: {source_classes}
        target_classes: {target_classes}
      ''')

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
