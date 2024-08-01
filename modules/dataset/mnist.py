import argparse
import torchvision
import torch
import PIL

from pathlib import Path

from modules.dataset.dataset_template import DatasetTemplate
from torchvision.datasets.mnist import read_image_file, read_label_file


class MNIST(DatasetTemplate):
    def __init__(self, root_path, data_config, mode, download=False):
        super(MNIST, self).__init__(root_path, data_config, mode, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = img.squeeze()  # Remove channel dimension if it is there
            img = PIL.Image.fromarray(img.numpy(), mode='L')  # Specify mode 'L' for grayscale

            img = self.transform(img)

        data_dict = {
            'img': img,
            'target': target,
        }
        return data_dict

    
    @staticmethod
    def download_data(data_path):
        Path(data_path).mkdir(parents=True, exist_ok=True)
        print('Downloading Training Data...')
        torchvision.datasets.MNIST(root=data_path, train=True, download=True)
        print('Downloading Testing Data...')
        torchvision.datasets.MNIST(root=data_path, train=False, download=True)
        print('MNIST Dataset Successfully Downloaded!')

    def load_data(self):
        raw_path = Path(self.data_path)
        image_file = f"{'train' if self.mode == 'train' else 't10k'}-images-idx3-ubyte"
        data = read_image_file(raw_path / image_file)

        label_file = f"{'train' if self.mode == 'train' else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(raw_path / label_file)

        return data, targets
        
    def __repr__(self):
        return "MNIST(root='%s', train=%s)" % (self.root, self.train)


def __main__():
    # Download and load the MNIST dataset
    parser = argparse.ArgumentParser(description="MNIST dataset loader")
    parser.add_argument("--data_path", default="./data/", type=str, help="Root directory of dataset")
    
    args = parser.parse_args()

    MNIST.download_data(args.data_path)

if __name__ == "__main__":
    __main__()