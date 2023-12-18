import os

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from util.Canny import GaussianBlur

import keras.backend as K
import tensorflow as tf

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges))*0 + (w3 * K.mean(l_depth)*10)

# %%
def gradient(image):
    # image = Grayscale(image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    G_x = cv2.Sobel(image, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    G_y = cv2.Sobel(image, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    return G_x.astype('int32'),G_y.astype('int32')

DATA_DIR = './data'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

# %%

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# %% md

# Dataloader and utility functions

# %%

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
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
    DIST_SCALES = ['0','1','2','3','4','5']

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
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.gradient_dir_values = [val for val in range(1,len(self.GRADIENT_DIRECTION))]
        self.dist_values = [val for val in range(len(self.DIST_SCALES))]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[50:-50,80:-80]
        ori_mask = cv2.imread(self.masks_fps[i], 0)
        gaussian_mask = GaussianBlur(ori_mask)
        G_x,G_y = gradient(gaussian_mask)
        angles = (np.rad2deg(np.arctan2(G_y, G_x)).astype('int8')+180)
        gradient_mask = np.around(angles)/360

        gradient_mask = np.stack([gradient_mask], axis=-1).astype('float')

        binary_mask = 1- ori_mask/255
        dist_mask = cv2.distanceTransform(binary_mask.astype('uint8'), cv2.DIST_L2, 3)
        dist_mask[np.where(dist_mask>=50)]=50
        dist_mask = np.around(dist_mask)/50
        dist_mask = np.stack([dist_mask], axis=-1).astype('float')

        ori_masks = [ori_mask/255]
        ori_mask = np.stack(ori_masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=np.concatenate((ori_mask, gradient_mask, dist_mask), axis=-1))
            image, mask = sample['image'], sample['mask']
            sep_index = ori_mask.shape[2]
            ori_mask = mask[:,:,0:sep_index]
            gradient_mask = mask[:,:,sep_index:sep_index+gradient_mask.shape[2]]
            sep_index = sep_index+gradient_mask.shape[2]
            dist_mask = mask[:, :, sep_index:sep_index + dist_mask.shape[2]]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=np.concatenate((ori_mask, gradient_mask, dist_mask), axis=-1))
            image, mask = sample['image'], sample['mask']
            sep_index = ori_mask.shape[2]
            ori_mask = mask[:,:,0:sep_index]
            gradient_mask = mask[:,:,sep_index:sep_index+gradient_mask.shape[2]]
            sep_index = sep_index+gradient_mask.shape[2]
            dist_mask = mask[:, :, sep_index:sep_index + dist_mask.shape[2]]

        return image, [ori_mask, gradient_mask, dist_mask]

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
        samples = [samples for samples in zip(*data)]
        data_batch = np.stack(samples[0], axis=0)
        mask_batch = [np.stack(c, axis=0) for c in zip(*samples[1])]
        batch =[data_batch,mask_batch]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

        # %%

CLASS = ['vessel']

### Augmentations

# %%

import albumentations as A


# %%

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),

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
        # A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
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

# %%

import segmentation_models as sm

# %%

BACKBONE = 'densenet121'
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 100

preprocess_input = sm.get_preprocessing(BACKBONE)

# %%

# define network parameters
n_classes = 1 if len(CLASS) == 1 else (len(CLASS) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.DODN(BACKBONE, classes=n_classes, activation=activation)

# %%

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
class_loss = dice_loss + (1 * focal_loss)
gradient_loss = depth_loss_function

dist_loss = depth_loss_function

model.compile(optim, loss={'cls_loss':class_loss, 'angle_loss': gradient_loss, 'dist_loss': dist_loss})

# %%

# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASS,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASS,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1][0].shape == (BATCH_SIZE, 320, 320, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./output/DODN.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# %%

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)
