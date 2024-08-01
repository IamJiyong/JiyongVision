"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import sys
import os
import pickle
import torch
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from pathlib import Path


from modules.dataset.dataset_template import DatasetTemplate


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None, keep_difficult=False, classes=None):
        self.class_to_ind = class_to_ind or dict(
            zip(classes, range(len(classes))))
        self.keep_difficult = keep_difficult


    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOC0712(DatasetTemplate):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root_path, data_config, mode, download=False):
        super(VOC0712, self).__init__(root_path, data_config, mode, download=download)
        self.target_transform = VOCAnnotationTransform(class_to_ind=self.class_to_ind, classes=self.class_names)


    def load_data(self):
        self._annopath = Path('%s', 'Annotations', '%s.xml')
        self._imgpath = Path('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in [('2007', 'trainval'), ('2012', 'trainval')]:
            rootpath = self.data_path / Path('VOC' + year)
            for line in open(rootpath / Path('ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        return self.ids, None
    

    @staticmethod
    def download_data(data_path):
        import os
        import urllib.request
        import urllib.request
        import tarfile

        if data_path.split('/')[-1] == 'VOCdevkit':
            data_dir = Path(data_path).parent
        else:
            data_dir = Path(data_path)
        data_dir.mkdir(parents=True, exist_ok=True)

        url_dict = {
            'VOCtrainval_06-Nov-2007': "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            'VOCtrainval_11-May-2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'VOCtest_06-Nov-2007': "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
        }
        
        for dataset_name, url in url_dict.items():
            print('Downloading %s ...' % dataset_name)
            download_path = data_dir / Path(url.split('/')[-1])
            urllib.request.urlretrieve(url, str(download_path))

            print('Extracting %s ...' % dataset_name)
            with tarfile.open(download_path, 'r') as tar_ref:
                tar_ref.extractall(str(data_dir))
            
            print('Removing %s.tar file ...' % dataset_name)
            os.remove(download_path)

        print('Download done!')


    def __getitem__(self, index):
        img, target, H, W = self.pull_item(index)

        data_dict = {
            'img_id': self.ids[index],
            'img': img,
            'gt_boxes': target,
            'original_size': (H, W)
        }

        return data_dict


    def __len__(self):
        return len(self.ids)


    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(str(self._annopath) % img_id).getroot()
        img = cv2.imread(str(self._imgpath) % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt
    

    @staticmethod
    def parse_rec(filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                                int(bbox.find('ymin').text) - 1,
                                int(bbox.find('xmax').text) - 1,
                                int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)

        return objects
    

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    

    def evaluate(self,
                 detpath,
                 class_name,
                 ovthresh=0.5,
                 use_07_metric=True):
        """rec, prec, ap = voc_eval(detpath,
                            annopath,
                            imagesetfile,
                            classname,
                            [ovthresh],
                            [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
    annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
        cachedir = Path(self.data_path) / 'annotations_cache'
        cachedir.mkdir(parents=True, exist_ok=True)
        cachefile = cachedir / (class_name + '_annots.pkl')
        # read list of images
        if cachefile.exists():
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
        else:
            recs = {}
            for i, img_id in enumerate(self.ids):
                recs[img_id[1]] = self.parse_rec(str(self._annopath) % img_id)
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(self.ids)))
            # save
            print('Saving cached annotations to {:s}'.format(str(cachefile)))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for _, img_id in self.ids:
            R = [obj for obj in recs[img_id] if obj['name'] == class_name]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[img_id] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if any(lines):

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = 0.

        return rec, prec, ap


def __main__():
    import argparse
    # Download and load the VOC dataset
    parser = argparse.ArgumentParser(description="VOC dataset loader")
    parser.add_argument("--data_path", default="./data/VOC0712", type=str, help="Root directory of dataset")
    
    args = parser.parse_args()

    VOC0712.download_data(args.data_path)

if __name__ == "__main__":
    __main__()