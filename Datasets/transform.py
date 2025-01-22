import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=20, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur (),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                #mean = (0.485, 0.456, 0.406),
                #std = (0.229, 0.224, 0.225),
                mean = (0., 0., 0.),
                std = (1., 1., 1.),
                max_pixel_value = 255.0
            ),
            ToTensorV2(),
        ])
val_transform = A.Compose(
    [
        #A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean = (0., 0., 0.),
            std = (1., 1., 1.),
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ])