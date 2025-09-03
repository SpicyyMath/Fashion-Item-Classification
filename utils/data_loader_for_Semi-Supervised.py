import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict, Union

class FashionSuperclassDataset(Dataset):
    """
    Fashion Big Categories dataset (semi-supervised learning)
    Function:
    - Handles only 4 superclasses of classification (shoes, clothing, bags, accessories)
    - Supports mixed loading of labelled and unlabelled data
    - Automatically convert small labels to large labels.
    - Provide data enhancement support
    """
    # Configuration of superclasses
    SUPERCLASS_MAP = {
        'shoes': 0,
        'clothing': 1,
        'bags': 2,
        'accessories': 3
    }
    # Small Class to Large Class Mapping
    SUBCLASS_TO_SUPERCLASS = {
        # bags
        'Back': 'bags', 'Clutch': 'bags',
        'CrossBody': 'bags', 'Shoulder': 'bags', 'TopHandle': 'bags', 'Tote': 'bags',      
        # shoes
        'Boots': 'shoes', 'Flats': 'shoes', 'Heels': 'shoes',
        'Sandals': 'shoes', 'Sneakers': 'shoes', 'Mules': 'shoes',
        # clothing
        'Dress': 'clothing', 'Top': 'clothing', 'Skirt': 'clothing',  'Trouser': 'clothing', 
        'Legwear': 'clothing','Pants': 'clothing', 'Outwear': 'clothing', 'Jumpsuit': 'clothing',
        # accessories
        'Belts': 'accessories', 'Bracelet': 'accessories', 'Earring': 'accessories','Rings': 'accessories',
        'Necklace': 'accessories', 'Watches': 'accessories', 'Sunglasses': 'accessories',
        'Eyewear': 'accessories', 'Hairwear': 'accessories','Hat': 'accessories', 
        'Gloves': 'accessories','Brooch': 'accessories'
    }
    # Standardised category name
    CLASS_STANDARDIZATION = {
         'Bracelets': 'Bracelet',
         'Dresses': 'Dress',
         'Earrings': 'Earring',
         'Hats': 'Hat',
         'Jumpsuits': 'Jumpsuit',
         'Necklaces': 'Necklace',
         'Skirts': 'Skirt',
         'Tops': 'Top',
         'Earing': 'Earring',
         'Neckline': 'Necklace',
         'Neckwear': 'Necklace',
         'Trousers': 'Trouser',
         'Crossbody': 'CrossBody',  
         'Tophandle': 'TopHandle',
         'unlabeled': None  
    }

    def __init__(self, image_dir: str, label_path: Optional[str] = None, transform: Optional[A.Compose] = None):
        """
        image_dir:String type specifying the path to the directory where the image file is located
        label_path:Optional string type, path to label file (can be None for no label data)
        transform:Compose type using the albumentations library
        """

        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform if transform else self.get_default_transform()
        
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        #If a label path is provided (label_path is not None), 
        #call the _load_labels() method to load the labels
        self.labels = self._load_labels() if label_path else None
        self.valid_indices = self._filter_valid_images()
    
    def _filter_valid_images(self) -> List[int]:
        """Filters and returns an indexed list of all valid images"""
        valid_indices = []
        for idx, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.image_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    img.convert('RGB')
                valid_indices.append(idx)
            except Exception as e:
                print(f"Ignore invalid images {img_file}: {str(e)}")
        return valid_indices
    
    
    #Load and process the tag file, return a list of tags corresponding to the image file
    def _load_labels(self) -> List[Optional[int]]:
        """
        Processing Flow:
         1.Check if the labels file exists → report error if it does not exist
         2.Read JSON file → get dictionary labels_data 
         3.Normalise filenames and parse label values
         4.Find corresponding labels for each image file
        """
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file does not exist: {self.label_path}")
        with open(self.label_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        label_mapping = {
            self._standardize_image_name(k): self._parse_label(v) 
            for k, v in labels_data.items()
        }
        return [label_mapping.get(self._standardize_image_name(f)) for f in self.image_files]
    
    def _standardize_image_name(self, name: str) -> str:
        return os.path.splitext(name.lower())[0]
    

    #Parses raw labels into a uniform superclass label ID (returns None for no labels)
    def _parse_label(self, label: Union[str, int]) -> Optional[int]:
        #digital processing
        if isinstance(label, str) and label.isdigit():
           label_int = int(label)
           return label_int if label_int in self.SUPERCLASS_MAP.values() else None
        if isinstance(label, int):
            return label if label in self.SUPERCLASS_MAP.values() else None
        #strings Processing
        if isinstance(label, str) and label.lower() == 'unlabeled':
           return None
        #Handling of small class labels
        class_name = self.CLASS_STANDARDIZATION.get(label, label)
        if class_name is None:
            return None
        superclass = self.SUBCLASS_TO_SUPERCLASS.get(class_name)
        if superclass is None:
            return None
        return self.SUPERCLASS_MAP[superclass]
    
    def get_default_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_augmentation(aug_level: str = 'medium') -> A.Compose:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if aug_level == 'weak':
            return A.Compose([
               A.Resize(height=256, width=256),
               A.RandomCrop(height=224, width=224,p=0.1),#Random cropping of 224x224 area with 50% probability
               A.HorizontalFlip(p=0.5),  # Flip the image horizontally with 50% probability
               A.ToGray(p=0.1),  #10% probability of conversion to grey scale image
               A.Normalize(mean=mean, std=std),
               ToTensorV2()
            ])
        elif aug_level == 'medium':
            return A.Compose([
                A.Resize(height=256, width=256),
                A.RandomResizedCrop(size=(256, 256),p=0.3),  
                A.HorizontalFlip(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),  #30% probability of adjusting brightness, contrast and saturation
                A.ToGray(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        elif aug_level == 'strong':
            return A.Compose([
                 A.Resize(height=256, width=256),
                 A.RandomResizedCrop(size=(256, 256),p=0.5),  
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5), # Stronger colour perturbation
                 A.Rotate(limit=30, p=0.5), # Random rotation (-30 degrees to 30 degrees)
                 A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5), # Random panning and zooming
                 A.ToGray(p=0.3), 
                 A.Normalize(mean=mean, std=std),
                 ToTensorV2()
            ])
        else:
            raise ValueError(f"Unknown level of data enhancement: {aug_level}")

    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        - image: image tensor
        - has_label: boolean with or without label (converted to int tensor)
        - label: label (-1 if no label)
        """
        actual_idx = self.valid_indices[idx]
        img_file = self.image_files[actual_idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        if self.labels is not None:
            label = self.labels[actual_idx]
            has_label = int(label is not None)  
            label = label if has_label else -1  
        else:
            has_label = 0
            label = -1
        
        return {
            'image': img,
            'has_label': torch.tensor(has_label, dtype=torch.int32),
            'label': torch.tensor(label, dtype=torch.int64)
        }
    
    def get_supervised_indices(self) -> List[int]:
        if self.labels is None:
            return []
        return [i for i, idx in enumerate(self.valid_indices) 
                if self.labels[idx] is not None]
    
    def get_unsupervised_indices(self) -> List[int]:
        if self.labels is None:
            return list(range(len(self.valid_indices)))
        return [i for i, idx in enumerate(self.valid_indices) 
               if self.labels[idx] is None]
    
    def prepare_data_lists(self) -> Tuple[List[torch.Tensor], List[bool], List[Optional[int]]]:
        """
        Returns a data structure that matches the requirements:
        - all_images: A list of all image tensors.
        - has_label: boolean list indicating whether there is a label or not
        - actual_labels: list of labels (None if no labels)
        """
        all_images = []
        has_label = []
        actual_labels = []
        
        for idx in self.valid_indices:
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)
            
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            if self.transform:
                img = self.transform(image=img)['image']
            
            all_images.append(img)
            
            if self.labels is not None:
                label = self.labels[idx]
                has_label.append(label is not None)
                actual_labels.append(label)
            else:
                has_label.append(False)
                actual_labels.append(None)
        
        return all_images, has_label, actual_labels


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batch data of dictionary type
    """
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'has_label': torch.stack([item['has_label'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def create_dataloaders(
    image_dir: str,
    label_path: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    augmentation: Optional[str] = None,
    **kwargs
) -> DataLoader:
    transform = FashionSuperclassDataset.get_augmentation(augmentation) if augmentation else None
    
    dataset = FashionSuperclassDataset(
        image_dir=image_dir,
        label_path=label_path,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  
        **kwargs
    )
    
    return loader


def create_semisupervised_dataloaders(
    image_dir: str,
    label_path: str,
    batch_size: int = 32,
    sup_batch_ratio: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
    augmentation: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    transform = FashionSuperclassDataset.get_augmentation(augmentation) if augmentation else None
    
    dataset = FashionSuperclassDataset(
        image_dir=image_dir,
        label_path=label_path,
        transform=transform
    )
    
    sup_indices = dataset.get_supervised_indices()
    unsup_indices = dataset.get_unsupervised_indices()
    
    sup_batch_size = max(1, int(batch_size * sup_batch_ratio))
    unsup_batch_size = batch_size - sup_batch_size
    
    sup_loader = DataLoader(
        dataset,
        batch_size=sup_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(sup_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )
    
    unsup_loader = DataLoader(
        dataset,
        batch_size=unsup_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(unsup_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )
    
    return sup_loader, unsup_loader


if __name__ == "__main__":
    
    # Create a dataset instance
    transform = FashionSuperclassDataset.get_augmentation('medium')
    dataset = FashionSuperclassDataset(
        image_dir="Data/images",
        label_path="Data/labels.json",
        transform=transform
    )
    # Total number of images tested
    all_images, has_label, actual_labels = dataset.prepare_data_lists()
    print("\nTotal number of images tested:")
    print(f"Total number of images: {len(all_images)}")
    print(f"Number of labeled images: {sum(has_label)}")
    print(f"The label of the first image: {actual_labels[0]}")
    print(f"Whether the first image has a label: {has_label[0]}")
    print(f"The first image tensor shape: {all_images[0].shape}")
    
    # DataLoader Functional Test
    mixed_loader = create_dataloaders(
        image_dir="Data/images",
        label_path="Data/labels.json",
        batch_size=4,
        augmentation='medium',
        num_workers=0 
    )
    
    batch = next(iter(mixed_loader))
    print("\nHybrid Data Loader Tests:")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Whether there is a label: {batch['has_label']}")
    print(f"Label: {batch['label']}")
    