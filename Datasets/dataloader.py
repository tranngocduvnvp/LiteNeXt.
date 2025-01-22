from Datasets.dataset import MedicalDataset
import torch.utils.data as data
from Datasets.transform import train_transform, val_transform
import multiprocessing
from sklearn.model_selection import train_test_split
import glob

def get_dataloader_lung(root, batch_size = 4):
    
    DATA_NAME = ["bowl2018"]
    list_img_paths = [glob.glob(f"{root}/{name}/img/*") for name in DATA_NAME]
    list_mask_paths = [glob.glob(f"{root}/{name}/mask/*") for name in DATA_NAME]
    
    train_img = []
    val_img = []
    test_img = []
    train_mask = []
    val_mask = []
    test_mask = []

   
    for img_dataset, mask_dataset in zip(list_img_paths, list_mask_paths):
        
        labels = [int(file.split("_")[-1].split(".")[0]) for file in img_dataset]
        img_train_val, img_test, mask_train_val, mask_test = train_test_split(img_dataset, mask_dataset, test_size=0.1, stratify=labels, random_state=42)
        labels = [int(file.split("_")[-1].split(".")[0]) for file in img_train_val]
        img_train, img_val, mask_train, mask_val = train_test_split(img_train_val, mask_train_val, test_size=1 - 0.8/0.9, stratify=labels, random_state=42)

        # Gộp vào các list tương ứng
        train_img.extend(img_train)
        val_img.extend(img_val)
        test_img.extend(img_test)
        train_mask.extend(mask_train)
        val_mask.extend(mask_val)
        test_mask.extend(mask_test)

   
    train_data = MedicalDataset(train_img,train_mask,transform=train_transform)
    val_data = MedicalDataset(val_img,val_mask,transform=val_transform)
    test_data = MedicalDataset(test_img,test_mask,transform=val_transform)

    train_dataloader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=multiprocessing.Pool()._processes,
    )
    
    val_dataloader = data.DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )
    
    test_dataloader = data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = get_dataloader_lung("LiteNeXt/dataset")
    print("len data train:", len(train_dataloader.dataset))
    print("len data val:", len(val_dataloader.dataset))
    print("len data test:", len(test_dataloader.dataset))