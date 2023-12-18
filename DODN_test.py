import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
import albumentations as A
from eval.prf_metrics import cal_prf_metrics
from eval.segment_metrics import cal_semantic_metrics
from util.graph_cut_interface import graph_cut_interface
from skimage.morphology import skeletonize,medial_axis
from scipy.ndimage import generic_filter
import matlab.engine
from scipy.interpolate import SmoothBivariateSpline
from util.Canny import Grayscale,GaussianBlur
import codecs
from scipy.io import savemat
from util.Canny import Canny
from PIL import Image

DATA_DIR = './data'
def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return 255 * ((P[4]==255) and np.sum(P)==510)

def clean(I):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(I.astype('int8'), connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    min_size = 20
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            I[output == i + 1] = 0
    """Central pixel and just one other must be set to be a line end"""
    return I

def gradient(image):
    # image = Grayscale(image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    G_x = cv2.Sobel(image, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    G_y = cv2.Sobel(image, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    return G_x,G_y

def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (angles[i, j] == 0) or (angles[i, j] == 4):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (angles[i, j] == 1) or (angles[i, j] == 5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (angles[i, j] == 2) or (angles[i, j] == 6):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed
# %%

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# helper function for data visualization
def visualize(file_name,**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(file_name)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['bg', 'vessel']
    GRADIENT_DIRECTION = ['bg', '0', '45', '90', '135', '180', '-135', '-90', '-45']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.gradient_dir_values = [val for val in range(1,len(self.GRADIENT_DIRECTION))]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        ### gt pixel is 255
        mask = mask/255
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480)
        A.PadIfNeeded(480, 640)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

BACKBONE = 'densenet121'
BATCH_SIZE = 8
CLASSES = ['vessel']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# %%

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.DODN(BACKBONE, classes=n_classes, activation=activation)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=len(test_dataset), shuffle=False)

model.load_weights('.output/DODN.h5')
# %%

n = len(test_dataset)
ids = range(0,n)
thresh_step = 0.01
pred_list, gt_list = [], []
for i in ids:
    image, gt_mask = test_dataset[i]
    index = test_dataset.images_fps[i].rindex('/')
    file_name = test_dataset.images_fps[i][index:-4]
    gt_mask = gt_mask.squeeze()
    gt_list.append(gt_mask)
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)
    pr_mask = pred_mask[0].squeeze()
    mdic = {"pr_mask": pr_mask.astype('float')}
    if not os.path.exists('./output_'+BACKBONE+'_mat'):
        os.mkdir('./output_'+BACKBONE+'_mat')
    savemat('./output_'+BACKBONE+'_mat' + file_name + '.mat', mdic)
    centerline = clean(skeletonize(pr_mask>0.1))
    pred_list.append(centerline)
    gradient_mask = pred_mask[1].squeeze()
    dist_mask = pred_mask[2].squeeze()
    dist_mask =dist_mask-np.min(dist_mask)
    dist_mask[np.where(dist_mask>0.6)]=0.6
    cv2.imwrite('./distance_map/' + test_dataset.ids[i], dist_mask * 255)
    angle_direction = gradient_mask*360-180
    angles = np.deg2rad(angle_direction)
    center_line = non_maximum_suppression(pr_mask, angle_direction)/255
    center_line = skeletonize(GaussianBlur(center_line)>0.01).astype('float')

    dist_masks = np.argmax(dist_mask,axis=-1)
    dist_mask = cv2.GaussianBlur(dist_masks.astype('float'), (25,25), 0)

    center_line_mask,center_line_refined = graph_cut_interface(center_line,dist_masks)
    center_line_refined = center_line_refined[:,:,0]/255
    center_line_refined = clean(center_line_refined)
    pred_list.append(center_line_refined)

output_path_prf = './prf/Unet_'+BACKBONE+'_4_4'+'.prf'
output_path_seg = './seg/Unet_'+BACKBONE+'_4_4'+'.seg'
final_prf_results = cal_prf_metrics(pred_list, gt_list, thresh_step)
final_prf_results_arr = np.array(final_prf_results)
print(final_prf_results_arr[49,3])

final_seg_results = cal_semantic_metrics(pred_list, gt_list, thresh_step)
final_seg_results_arr = np.array(final_seg_results)

with codecs.open(output_path_prf, 'w', encoding='utf-8') as fout:
    for ll in final_prf_results:
        line = '\t'.join(['%.4f' % v for v in ll]) + '\n'
        fout.write(line)
with codecs.open(output_path_seg, 'w', encoding='utf-8') as fout:
    for ll in final_seg_results:
        line = '\t'.join(['%.4f' % v for v in ll]) + '\n'
        fout.write(line)