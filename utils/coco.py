import os
import torch
import json
from torchvision.datasets import ImageFolder, CocoDetection
from torch.utils.data.dataloader import default_collate


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

_COCO_CLASS_TO_INDEX = {c: i for i, c in enumerate([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
])}


def coco_as_class_ids(label):
    """Convert a COCO detection label to the list of class IDs.

    Args:
        label (list of dict): an image label in the VOC detection format.

    Returns:
        list: List of ids of classes in the image.
    """
    if len(label) == 0:
        return []
    return list({_COCO_CLASS_TO_INDEX[ann['category_id']] for ann in label})


def coco_as_mask(dataset, label, class_id):
    """Convert a COCO detection label to a mask.

    Return a boolean mask for the regions of :attr:`class_id`.

    If the label is the empty list, because there are no objects at all in the
    image, the function returns ``None``.

    Args:
        label (array of dict): an image label in the VOC detection format.
        class_id (int): ID of the requested class.

    Returns:
        :class:`torch.Tensor`: 2D boolean tensor.
    """
    assert isinstance(class_id, int)
    mask = None
    if not label:
        return mask
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    for ann in label:
        this_class_id = _COCO_CLASS_TO_INDEX[ann['category_id']]
        if class_id == this_class_id:
            if mask is None:
                mask = torch.zeros(
                    image['height'],
                    image['width'],
                    dtype=torch.uint8
                )
            this_mask = dataset.coco.annToMask(ann)
            mask.add_(torch.tensor(this_mask))
    if mask is not None:
        mask = mask > 0
    mask = mask.to(torch.bool)
    return mask


def coco_as_image_size(dataset, label):
    """Convert a COCO detection label to the image size.

    Args:
        label (list of dict): an image label in the VOC detection format.

    Returns:
        tuple: width, height of image.
    """
    if not label:
        return None
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    return image['width'], image['height']


def coco_as_image_name(dataset, label):
    """Convert a COCO detection label to the image name.

    Args:
        label (list of dict): an image label in the COCO detection format.

    Returns:
        str: image name.
    """
    if not label:
        return None
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    return os.path.splitext(image['file_name'])[0]


class ImageFolder(ImageFolder):
    """Image folder dataset.

    This class extends :class:`torchvision.datasets.ImageFolder`.
    Its constructor supports the following additional arguments:

    Args:
        limiter (int, optional): limit the dataset to :attr:`limiter` images,
            picking from each class in a round-robin fashion.
            Default: ``None``.
        full_classes (list of str, optional):  list of full class names.
            Default: ``None``.

    Attributes:
        selection (list of int): indices of the active images.
        full_classes (list of str): class names.
    """

    def __init__(self, *args, limiter=None, full_classes=None, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)
        num_images = super(ImageFolder, self).__len__()
        self.selection = range(num_images)
        self.full_classes = full_classes
        if not limiter:
            return
        # Pick one sample per class in a round-robin manner.
        class_indices = [
            [i for i, y in enumerate(self.targets) if y == label]
            for label in range(len(self.classes))
        ]
        triplets = [
            (k, y, i)
            for y, indices in enumerate(class_indices)
            for k, i in enumerate(indices)
        ]
        triplets.sort()
        self.selection = [i for k, y, i in triplets[:min(limiter, num_images)]]
        self.selection = sorted(self.selection)

    def __getitem__(self, index):
        return super().__getitem__(self.selection[index])

    def __len__(self):
        return len(self.selection)

    def get_image_url(self, i):
        return self.samples[self.selection[i]][0]


class CocoDetection(CocoDetection):
    """COCO Detection dataset.
    The data can be downloaded at `<http://cocodataset.org/#download>`__.
    Args:
        limiter (int, optional): limit the dataset to the first :attr:`limiter`
            images. Default: ``None``.

    Attributes:
        classes (list of str): class names.
        selection (list of int): indices of the active images.
    """

    def __init__(self, root, annFile, *args, limiter=None, **kwargs):
        super(CocoDetection, self).__init__(root, annFile, *args, **kwargs)
        self.subset = os.path.splitext(os.path.basename(annFile))[0]
        num_images = super(CocoDetection, self).__len__()
        if limiter:
            num_images = min(num_images, limiter)
        self.selection = range(num_images)
        self.classes = COCO_CLASSES

    def __getitem__(self, index):
        return super().__getitem__(self.selection[index])

    def __len__(self):
        return len(self.selection)

    def get_image_url(self, i):
        i = self.selection[i]
        image_id = self.ids[i]
        return self.coco.loadImgs(image_id)[0]['file_name']

    @property
    def images(self):
        """list of str: paths to images."""
        return [self.coco.loadImgs(i)[0]['file_name'] for i in self.ids]

    def as_class_ids(self, label):
        """Convert a label to list of class IDs.

        The same as :func:`coco_as_class_ids`.
        """
        return coco_as_class_ids(label)

    def as_mask(self, label, class_id):
        """Convert a label to a mask.

        The same as :func:`coco_as_mask`.
        """
        return coco_as_mask(self, label, class_id)

    def as_image_size(self, label):
        """Convert a label to the image size.

        The same as :func:`coco_as_image_size`.
        """
        return coco_as_image_size(self, label)

    def as_image_name(self, label):
        """Convert a label to the image name.

        The same as :func:`coco_as_image_name.`.
        """
        return coco_as_image_name(self, label)

    @staticmethod
    def collate(batch):
        """Collate function for use in a data loader."""
        inputs = default_collate([elem[0] for elem in batch])
        labels = [elem[1] for elem in batch]
        return inputs, labels


def get_dataset(subset="val2017",
                dataset_dir="/vcu/data/coco2017",
                annotation_dir="/vcu/data/coco2017/annotations",
                transform=None,
                limiter=None):

    im_path = os.path.join(dataset_dir, subset)
    ann_path = os.path.join(annotation_dir,
                            'instances_{}.json'.format(subset))
    dataset = CocoDetection(im_path,
                            ann_path,
                            transform=transform,
                            limiter=limiter)

    return dataset