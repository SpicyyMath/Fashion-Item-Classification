import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from PIL import Image
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import defaultdict
import traceback
from itertools import chain

# subclass to index
class_to_idx = {
    'Back': 0, 'Bags': 1, 'Clutch': 2, 'CrossBody': 3, 
    'Shoulder': 4, 'TopHandle': 5, 'Tote': 6,
    'Belts': 7, 'Boots': 8, 'Bracelet': 9, 'Brooch': 10,
    'Dress': 11, 'Earring': 12, 'Eyewear': 13, 'Flats': 14,
    'Gloves': 15, 'Hairwear': 16, 'Hat': 17, 'Heels': 18,
    'Jumpsuit': 19, 'Legwear': 20, 'Mules': 21, 'Necklace': 22,
    'Outwear': 23, 'Pants': 24, 'Rings': 25, 'Sandals': 26,
    'Skirt': 27, 'Sneakers': 28, 'Sunglasses': 29, 'Top': 30,
    'Trouser': 31, 'Watches': 32, 'Shoes': 33
}

# subclass to superclass
subclass_to_superclass = {
    'Heels': 'shoes', 'Boots': 'shoes', 'Flats': 'shoes',
    'Sandals': 'shoes', 'Sneakers': 'shoes', 'Mules': 'shoes', 'Shoes': 'shoes',
    'Skirt': 'clothing', 'Dress': 'clothing', 'Top': 'clothing',
    'Trouser': 'clothing', 'Jumpsuit': 'clothing', 'Outwear': 'clothing',
    'Legwear': 'clothing', 'Pants': 'clothing',
    'Back': 'bags', 'Bags': 'bags', 'Clutch': 'bags',
    'CrossBody': 'bags', 'Shoulder': 'bags', 'TopHandle': 'bags',
    'Tote': 'bags',
    'Earring': 'accessories', 'Necklace': 'accessories',
    'Watches': 'accessories', 'Rings': 'accessories',
    'Eyewear': 'accessories', 'Hairwear': 'accessories',
    'Hat': 'accessories', 'Gloves': 'accessories',
    'Bracelet': 'accessories', 'Belts': 'accessories',
    'Brooch': 'accessories', 'Sunglasses': 'accessories'
}

# standardize class names
standardize_class_names = {
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
    'Tophandle': 'TopHandle'
}

# superclss to index
superclass_to_idx = {
    'shoes': 0,
    'clothing': 1,
    'bags': 2,
    'accessories': 3
}

num_classes = len(class_to_idx)
num_classes_superclass = len(superclass_to_idx)



class FashionDataset(Dataset):
    """自定义时尚数据集类（处理字典格式的labels.json）"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        
        # 加载labels文件夹中的labels.json
        labels_path = os.path.join(root_dir, 'labels', 'labels.json')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"标签文件不存在: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
            
            # 将字典格式转换为代码预期的列表格式
            self.all_labels = [
                {"image_name": f"{img_id}.jpg", "class": class_name}  
                for img_id, class_name in labels_dict.items()
            ]
        
        # 获取所有图像文件并建立映射（不区分大小写）
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 创建图像文件名到标签的映射
        self.label_map = {
            item['image_name'].lower(): item  # 使用小写作为键
            for item in self.all_labels
        }
        
        # 按类别组织样本索引
        self.class_indices = defaultdict(list)
        for idx, img_name in enumerate(self.image_files):
            label_data = self.label_map.get(img_name.lower())
            if label_data:
                class_name = standardize_class_names.get(label_data['class'], label_data['class'])
                if class_name in class_to_idx:
                    self.class_indices[class_name].append(idx)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 查找标签（不区分大小写）
        label_data = self.label_map.get(img_name.lower())
        if label_data is None:
            available_files = "\n".join(list(self.label_map.keys())[:5])
            raise ValueError(
                f"找不到图像 {img_name} 对应的标签数据\n"
                f"前5个可用标签键:\n{available_files}"
            )
        
        # 标准化类别名称并转换为索引
        class_name = standardize_class_names.get(label_data['class'], label_data['class'])
        try:
            label = class_to_idx[class_name]
        except KeyError:
            raise ValueError(
                f"未知类别: {label_data['class']}\n"
                f"已知类别: {list(class_to_idx.keys())}"
            )
        
        # 加载图像（自动处理不同图像格式）
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)  # 转换为numpy数组供albumentations使用
        except Exception as e:
            raise ValueError(f"无法加载图像 {img_path}: {str(e)}")
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

class FashionDataset2(Dataset):
    """自定义时尚数据集类（处理字典格式的labels.json）"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        
        # 加载labels文件夹中的labels.json
        labels_path = os.path.join(root_dir, 'labels', 'labels.json')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"标签文件不存在: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
            
            # 将字典格式转换为代码预期的列表格式
            self.all_labels = [
                {"image_name": f"{img_id}.jpg", "class": class_name}  
                for img_id, class_name in labels_dict.items()
            ]
        
        # 获取所有图像文件并建立映射（不区分大小写）
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 创建图像文件名到标签的映射
        self.label_map = {
            item['image_name'].lower(): item  # 使用小写作为键
            for item in self.all_labels
        }
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 查找标签（不区分大小写）
        label_data = self.label_map.get(img_name.lower())
        if label_data is None:
            available_files = "\n".join(list(self.label_map.keys())[:5])
            raise ValueError(
                f"找不到图像 {img_name} 对应的标签数据\n"
                f"前5个可用标签键:\n{available_files}"
            )
        
        # 标准化类别名称并转换为大类索引
        class_name = standardize_class_names.get(label_data['class'], label_data['class'])
        try:
            # 先获取小类对应的大类，再获取大类的索引
            superclass = subclass_to_superclass[class_name]
            label = superclass_to_idx[superclass]
        except KeyError:
            raise ValueError(
                f"未知类别: {label_data['class']}\n"
                f"已知小类: {list(class_to_idx.keys())}\n"
                f"已知大类: {list(superclass_to_idx.keys())}"
            )
        
        # 加载图像（自动处理不同图像格式）
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)  # 转换为numpy数组供albumentations使用
        except Exception as e:
            raise ValueError(f"无法加载图像 {img_path}: {str(e)}")
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_data_transforms(augmentation_level='medium'):
    """获取数据增强转换  
    参数:
        augmentation_level (str): 数据增强强度级别，可选'weak'、'medium'或'strong'
    返回:
        tuple: 包含训练集和验证/测试集的数据转换管道"""
    mean = [0.485, 0.456, 0.406] # RGB三通道的均值
    std = [0.229, 0.224, 0.225]  # RGB三通道的标准差
    
    val_test_transforms = A.Compose([
        A.Resize(height=256, width=256),  # 首先将图像调整为224x224像素
        A.Normalize(mean=mean, std=std), # 使用预定义的均值和标准差进行归一化
        ToTensorV2()
    ])
    
    if augmentation_level == 'weak':
        train_transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),  # 以50%概率水平翻转图像
            A.ToGray(p=0.5),  #10%概率转换为灰度图像
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    elif augmentation_level == 'medium':
        train_transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),  # 30%概率调整亮度、对比度和饱和度
            A.ToGray(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    elif augmentation_level == 'strong':
        train_transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5), # 更强的颜色扰动
            A.Rotate(limit=30, p=0.5), # 随机旋转(-30度到30度)
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5), # 随机平移和缩放
            A.ToGray(p=0.5), 
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return train_transforms, val_test_transforms
def create_fewshot_loaders(dataset, support_size=5, query_size=5, min_samples=10, batch_size=32, num_workers=4, merge_rare_classes=True):
    """
    少样本学习的数据加载器
    参数:
        dataset: FashionDataset实例
        support_size: 每个少样本类的支持集样本数
        query_size: 每个少样本类的查询集样本数
        min_samples: 定义少样本类的最小样本数阈值
        batch_size: 数据加载器的批量大小
        merge_rare_classes: 是否合并极少数样本的类别(样本数<3)
    返回:
        train_loader_large: 大样本类别的数据加载器
        support_loader: 少样本类别的支持集数据加载器
        query_loader: 少样本类别的查询集数据加载器
    """
    # 识别少样本类别
    fewshot_classes = [cls for cls, indices in dataset.class_indices.items() 
                      if len(indices) <= min_samples]
    
    print("\n=== 初始少样本类别 ===")
    print(f"检测到{len(fewshot_classes)}个少样本类别（样本数<={min_samples}）:")
    for cls in fewshot_classes:
        print(f"  {cls}: {len(dataset.class_indices[cls])}样本")

    # 合并极少数样本的类别到相似大类（可选）
    merged_classes = []
    if merge_rare_classes:
        for cls in fewshot_classes.copy():
            if len(dataset.class_indices[cls]) < 3:  # 样本数少于3的类别
                superclass = subclass_to_superclass.get(cls, None)
                if superclass:
                    # 找到同大类的其他类别
                    similar_classes = [c for c in subclass_to_superclass 
                                     if subclass_to_superclass[c] == superclass and c != cls]
                    if similar_classes:
                        # 合并到同大类的第一个类别
                        target_class = similar_classes[0]
                        dataset.class_indices[target_class].extend(dataset.class_indices[cls])
                        del dataset.class_indices[cls]
                        fewshot_classes.remove(cls)
                        merged_classes.append((cls, target_class))
    
        if merged_classes:
            print("\n=== 合并的极少数样本类别 ===")
            for src, target in merged_classes:
                print(f"  {src} -> {target} (原样本数: {len(dataset.class_indices[target]) - len(dataset.class_indices[src])})")
    
    # 更新少样本类别列表
    fewshot_classes = [cls for cls, indices in dataset.class_indices.items() 
                      if len(indices) <= min_samples]
    
    print("\n=== 最终少样本类别 ===")
    print(f"剩余{len(fewshot_classes)}个少样本类别:")
    for cls in fewshot_classes:
        print(f"  {cls}: {len(dataset.class_indices[cls])}样本")

    # 构建大样本类别数据集
    large_class_indices = []
    large_classes = []
    for cls, indices in dataset.class_indices.items():
        if cls not in fewshot_classes:
            large_class_indices.extend(indices)
            large_classes.append(cls)
    
    print("\n=== 大样本类别 ===")
    print(f"共{len(large_classes)}个类别，{len(large_class_indices)}个样本")

    # 构建少样本类别的支持集和查询集
    support_indices = []
    query_indices = []
    fewshot_stats = []
    
    for cls in fewshot_classes:
        indices = dataset.class_indices[cls]
        total_samples = len(indices)
        
        if total_samples < support_size + query_size:
         # 如果样本不足，优先保证查询集
           actual_query = min(query_size, total_samples)
           actual_support = max(0, total_samples - actual_query)
        else:
           actual_query = query_size
           actual_support = min(support_size, total_samples - query_size)
        
        if actual_support >= actual_query and actual_query > 0:
           # 如果支持集≥查询集，调整分配
           adjust = (actual_support - actual_query) + 1
           if actual_support - adjust >= 1:  # 确保支持集至少有1个样本
              actual_support -= adjust
              actual_query += adjust
    
        # 随机划分支持集和查询集
        np.random.shuffle(indices)
        cls_support = indices[:actual_support]
        cls_query = indices[actual_support:actual_support+actual_query]
    
        support_indices.extend(cls_support)
        query_indices.extend(cls_query)
        fewshot_stats.append((cls, total_samples, len(cls_support), len(cls_query)))

    print("\n=== 少样本类别划分详情 ===")
    print(f"{'类别':<15} {'总样本':<8} {'支持集':<8} {'查询集':<8}")
    for cls, total, sup, qry in fewshot_stats:
        print(f"{cls:<15} {total:<8} {sup:<8} {qry:<8}")

    # 样本总数验证
    total_allocated = len(large_class_indices) + len(support_indices) + len(query_indices)
    print(f"\n=== 样本分配验证 ===")
    print(f"总样本数: {len(dataset)}")
    print(f"已分配样本: {total_allocated}")
    if total_allocated != len(dataset):
        print(f"警告: {len(dataset) - total_allocated}个样本未被分配！")
        # 查找未分配的样本
        all_indices = set(range(len(dataset)))
        allocated = set(large_class_indices + support_indices + query_indices)
        unallocated = all_indices - allocated
        print(f"未分配样本索引示例: {sorted(unallocated)[:5]}...")

    # 创建数据加载器
    # 大样本类别的数据加载器（使用训练集的数据增强）
    train_transforms, _ = get_data_transforms(augmentation_level='medium')
    large_dataset = Subset(dataset, large_class_indices)
    large_dataset.dataset.transform = train_transforms  # 应用数据增强
    
    train_loader_large = DataLoader(
        large_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 支持集的数据加载器（轻微数据增强）
    support_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.3),  # 轻微增强
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    support_dataset = Subset(dataset, support_indices)
    support_dataset.dataset.transform = support_transforms
    
    support_loader = DataLoader(
        support_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 查询集的数据加载器（无数据增强）
    query_transforms = A.Compose([
        ToTensorV2()  # 仅转换张量，无任何增强
    ])
    query_dataset = Subset(dataset, query_indices)
    query_dataset.dataset.transform = query_transforms
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的批次
    )
    
    print("\n=== 数据加载器统计 ===")
    print(f"大样本训练集: {len(train_loader_large)}批次 (共{len(large_dataset)}样本)")
    print(f"支持集: {len(support_loader)}批次 (共{len(support_dataset)}样本)")
    print(f"查询集: {len(query_loader)}批次 (共{len(query_dataset)}样本)")
    
    return train_loader_large, support_loader, query_loader, fewshot_classes

def create_data_loaders(data_dir, batch_size=32, augmentation_level='medium', num_workers=4, fewshot=False,merge_rare_classes=True):
    """创建数据加载器（主接口）
    参数:
        data_dir: 数据目录路径
        batch_size: 批量大小
        augmentation_level: 数据增强级别
        fewshot: 是否启用少样本学习模式
    返回:
        如果fewshot为True，返回 (train_loader, val_loader, test_loader, train_loader_large, support_loader, query_loader, fewshot_classes)
        否则返回 (train_loader, val_loader, test_loader)
    """
    train_transforms, val_test_transforms = get_data_transforms(augmentation_level)
    
    train_dataset = FashionDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = FashionDataset(
        os.path.join(data_dir, 'val'),
        transform=val_test_transforms
    )
    
    test_dataset = FashionDataset(
        os.path.join(data_dir, 'test'),
        transform=val_test_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if fewshot:
        # 创建少样本学习的数据加载器
        train_loader_large, support_loader, query_loader, fewshot_classes = create_fewshot_loaders(
            train_dataset,
            support_size=5,
            query_size=5,
            min_samples=10,
            batch_size=batch_size,
            num_workers=num_workers,
            merge_rare_classes=merge_rare_classes #添加merge_rare_classes参数控制是否合并极少数样本的类别
        )
        
        return (train_loader, val_loader, test_loader, 
                train_loader_large, support_loader, query_loader, fewshot_classes)
    else:
        return train_loader, val_loader, test_loader

idx_to_class = {v: k for k, v in class_to_idx.items()}
# 新增大类映射
idx_to_superclass = {v: k for k, v in superclass_to_idx.items()}

def create_data_loaders2(data_dir, batch_size=32, augmentation_level='medium', num_workers=4):
    """创建数据加载器（主接口）"""
    train_transforms, val_test_transforms = get_data_transforms(augmentation_level)
    
    train_dataset = FashionDataset2(
        os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = FashionDataset2(
        os.path.join(data_dir, 'val'),
        transform=val_test_transforms
    )
    
    test_dataset = FashionDataset2(
        os.path.join(data_dir, 'test'),
        transform=val_test_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_class_info(pred_idx):
    """根据预测索引获取类别信息"""
    subclass = idx_to_class[pred_idx]
    superclass = subclass_to_superclass.get(subclass, 'unknown')
    return {
        'subclass': subclass,
        'superclass': superclass,
        'display_text': f"这张图片是{superclass}中的{subclass}"
    }

if __name__ == "__main__":
    import traceback
    
    current_dir = Path(__file__).parent
    test_root = current_dir.parent / "Data"
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    try:
        # 检查大类映射完整性
        missing_superclasses = [k for k in class_to_idx if k not in subclass_to_superclass]
        if missing_superclasses:
           print(f"警告: 以下小类缺少大类映射: {missing_superclasses}")
        else:
           print("✓ 所有小类都有对应的大类映射")
        
        # 测试数据集加载
        train_dataset = FashionDataset(os.path.join(test_root, 'train'), transform=transform)
        val_dataset = FashionDataset(os.path.join(test_root, 'val'), transform=transform)
        test_dataset = FashionDataset(os.path.join(test_root, 'test'), transform=transform)
        
        print(f"✓ 数据集加载成功")
        print(f"总样本数: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        print(f"类别数: {num_classes}")
        
        # 检查第一个样本
        img, label = train_dataset[0]
        print(f"训练集样本0 | 图像形状: {img.shape} | 标签: {label} ({idx_to_class[label]})")
        
        # 验证标签有效性
        invalid_classes = []
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for item in dataset.all_labels:
                class_name = standardize_class_names.get(item['class'], item['class'])
                if class_name not in class_to_idx:
                    invalid_classes.append(item['class'])
        
        if not invalid_classes:
            print("✓ 所有标签映射有效")
        else:
            print(f"! 发现无效标签: {set(invalid_classes)}")
        
        # 检查图像尺寸范围
        print("\n训练集前5个样本的尺寸:")
        for i in range(5):
            img, _ = train_dataset[i]
            print(f"样本{i}: {img.shape}")
            
        # 测试数据加载器
        print("\n测试标准数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=test_root,
            batch_size=4,
            augmentation_level='medium',
            fewshot=False
        )
        print(f"训练集批次: {len(train_loader)} | 验证集批次: {len(val_loader)} | 测试集批次: {len(test_loader)}")
        
        # 测试少样本数据加载器
        print("\n测试少样本数据加载器...")
        train_loader, val_loader, test_loader, train_loader_large, support_loader, query_loader, fewshot_classes = create_data_loaders(
            data_dir=test_root,
            batch_size=4,
            augmentation_level='medium',
            fewshot=True
        )
        print(f"少样本类别: {fewshot_classes}")
        print(f"大样本训练集批次: {len(train_loader_large)}")
        print(f"支持集批次: {len(support_loader)} (样本数: {len(support_loader.dataset)})")
        print(f"查询集批次: {len(query_loader)} (样本数: {len(query_loader.dataset)})")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()