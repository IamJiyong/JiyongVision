import argparse
import PIL
import pickle
import numpy as np
import torchvision

from pathlib import Path

from modules.dataset.dataset_template import DatasetTemplate


class Cifar10(DatasetTemplate):
    def __init__(self, root_path, data_config, mode, download=False):
        super(Cifar10, self).__init__(root_path, data_config, mode, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = img.transpose((1, 2, 0))
            img = PIL.Image.fromarray(img)
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
        torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
        print('Downloading Testing Data...')
        torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
        print('Cifar10 Dataset Successfully Downloaded!')

    def load_data(self):
        if self.mode == 'train':
            data = []
            labels = []
            for i in range(1, 6):
                path = self.data_path / Path("data_batch_%d" % i)
                with open(path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                data.append(batch[b"data"])
                labels += batch[b"labels"]
            data = np.vstack(data).reshape(-1, 3, 32, 32)
            labels = np.array(labels)
        else:
            path = self.data_path / Path("test_batch")
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                data = batch[b"data"].reshape(-1, 3, 32, 32)
                labels = np.array(batch[b"labels"])

        return data, labels

    def __repr__(self):
        return "Cifar10(root='%s', train=%s)" % (self.root, self.train)
    

def __main__():
    # Download and load the CIFAR-10 dataset
    parser = argparse.ArgumentParser(description="CIFAR-10 dataset loader")
    parser.add_argument("--data_path", default="./data/CIFAR10", type=str, help="Root directory of dataset")
    
    args = parser.parse_args()

    Cifar10.download_data(args.data_path)

if __name__ == "__main__":
    import sys
    __main__()