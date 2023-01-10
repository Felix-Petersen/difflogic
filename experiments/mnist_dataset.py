import codecs
import os
import os.path
import warnings
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.error import URLError

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset


class MNISTRemoveBorderTransform:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        horizontal_black_lines = (image == 0.).all(dim=2)
        top_black = 0
        while horizontal_black_lines[0, top_black] and top_black < 14:
            top_black += 1
        bottom_black = 0
        while horizontal_black_lines[0, -bottom_black - 1] and bottom_black < 14:
            bottom_black += 1
        assert top_black + bottom_black >= 8, (top_black, bottom_black)
        while top_black + bottom_black >= 10:
            if top_black > 0:
                top_black -= 1
            if bottom_black > 0:
                bottom_black -= 1
        if top_black + bottom_black == 9:
            if top_black > 0:
                top_black -= 1
            else:
                bottom_black -= 1
        assert top_black + bottom_black == 8, (top_black, bottom_black)
        image = image[:, top_black:28 - bottom_black]

        vertical_black_lines = (image == 0.).all(dim=1)
        left_black = 0
        while vertical_black_lines[0, left_black] and left_black < 14:
            left_black += 1
        right_black = 0
        while vertical_black_lines[0, -right_black - 1] and right_black < 14:
            right_black += 1
        assert left_black + right_black >= 8, (left_black, right_black)
        while left_black + right_black >= 10:
            if left_black > 0:
                left_black -= 1
            if right_black > 0:
                right_black -= 1
        if left_black + right_black == 9:
            if left_black > 0:
                left_black -= 1
            else:
                right_black -= 1
        assert left_black + right_black == 8, (left_black, right_black)
        image = image[:, :, left_black:28 - right_black]

        return image


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        # 'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            remove_border=False
    ) -> None:
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = self._load_data()

        if remove_border:
            if transform is None:
                transform = torchvision.transforms.ToTensor()

            self.transform = torchvision.transforms.Compose([
                transform,
                MNISTRemoveBorderTransform(),
            ])
        else:
            if transform is None:
                self.transform = torchvision.transforms.ToTensor()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=True)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 3)
    return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    sqrt_of_n_imgs = 10

    # ds = MNIST('data-mnist', download=True, transform=torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     MNISTRemoveBorderTransform(),
    # ]))
    ds = MNIST('data-mnist', download=True, remove_border=True)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False))

    fig, ax = plt.subplots(nrows=sqrt_of_n_imgs, ncols=sqrt_of_n_imgs)
    ax = ax.flatten()
    for idx in range(sqrt_of_n_imgs ** 2):
        image, _ = next(loader)
        ax[idx].imshow(image.squeeze(1).squeeze(0).numpy())

    plt.show()

    print('Speed test...')

    ds = MNIST('data-mnist', download=True)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False))
    t_s = time.time()
    for _ in loader:
        pass
    t_e = time.time()
    print('regular', t_e - t_s)

    ds = MNIST('data-mnist', download=True, remove_border=True)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False))
    t_s = time.time()
    for _ in loader:
        pass
    t_e = time.time()
    print('remove_border', t_e - t_s)
