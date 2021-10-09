
import numpy as np
import itertools
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Cifar35200(Dataset):
    def __init__(self, split: str, data_path: str, transform=None, target_transform=None):

        # split: train, valid, test
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        data_path = Path(data_path)
        train_files = [f'data_batch_{i}.bin' for i in range(1,6)]
        valid_files = ['test_batch.bin']

        cifar_path = data_path / 'cifar-10-batches-bin'
        self.train_paths = [cifar_path / f for f in train_files]
        self.valid_paths = [cifar_path / f for f in valid_files]

        self.test_labels_path = data_path / 'cifar10.1_v6_labels.npy'
        self.test_images_path = data_path / 'cifar10.1_v6_data.npy'

        label_path = cifar_path / 'batches.meta.txt'
        label_names = [name for name in label_path.read_text().split('\n') if name]
        self.classes = label_names

        if self.split == 'test':
            images = np.load(self.test_images_path)
            labels = np.load(self.test_labels_path)
            self.data = tuple(zip(labels, images))
        elif self.split == 'valid':
            # https://stackoverflow.com/a/41289772
            shards = (tuple(self._load_data(p)) for p in self.valid_paths)
            self.data = list(itertools.chain.from_iterable(shards))
        else:
            shards = (tuple(self._load_data(p)) for p in self.train_paths)
            self.data = list(itertools.chain.from_iterable(shards))

    def _load_data(self, path):
        path = Path(path)
        data = path.read_bytes()
        offset = 0
        max_offset = len(data) - 1
        while offset < max_offset:
            labels = np.frombuffer(data, dtype=np.uint8, count=1, offset=offset).squeeze()
            offset += 1
            img = (np.frombuffer(data, dtype=np.uint8, count=3072, offset=offset).reshape((3, 32, 32)).transpose((1 ,2, 0)))
            offset += 3072
            yield labels, img

    def __getitem__(self, index):
        label, img = self.data[index]
        img = Image.fromarray(img)
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)

    def __len__(self):
        count = len(self.data)
        return count

def get_data_loaders(data_path='data', batch_size=32, num_workers=2, return_as_dict=True,
                     transform_train = None, transform_valid=None, transform_test=None):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),     #Rotates the image to a specified angel
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
        ])

    if transform_valid is None:
        transform_valid = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if transform_test is None:
        transform_test = transform_valid

    common_load_args = dict(batch_size=batch_size, num_workers=num_workers)

    train_set = Cifar35200('train', data_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **common_load_args)

    valid_set = Cifar35200('valid', data_path, transform=transform_valid)
    valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=False, **common_load_args)

    test_set = Cifar35200('test', data_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **common_load_args)

    if return_as_dict:
        return dict(
            train       = train_loader,
            validate    = valid_loader,
            test        = test_loader
        )
    else:
        return train_loader, valid_loader, test_loader