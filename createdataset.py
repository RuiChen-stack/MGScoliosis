import os
import torch
import torch.utils.data as data
import torch
import logging
from PIL import Image
from timm.data import create_parser

_ERROR_RETRY = 50
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)
_logger = logging.getLogger(__name__)

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root

class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
            num_classes=None,
            angel_num_classes=10
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.num_classes = num_classes
        self.angel_num_classes = angel_num_classes

    def __getitem__(self, index):
        img, target, path = self.parser[index]
        angel = path.split('/')[-1].split('-')[1]
        angel = int(angel)
        if angel>46:
            angel = 46
        angel = int(angel/5)
        label = torch.zeros(self.num_classes-1,2)
        angel_target = torch.zeros(self.angel_num_classes-1,2)
        label[:target] = torch.tensor([1,0])
        label[target:] = torch.tensor([0,1])
        angel_target[:angel] = torch.tensor([1,0])
        angel_target[angel:] = torch.tensor([0,1])
        target = label
        # print(angel_target)
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, angel_target
        # return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        repeats=0,
        **kwargs
):
    if search_split and os.path.isdir(root):
        root = _search_split(root, split)
    ds = ImageDataset(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
    return ds