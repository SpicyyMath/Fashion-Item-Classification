#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import shutil
import random

labels_path = r"C:\Users\18042\9444\labels.json"
with open(labels_path, "r") as f:
    labels = json.load(f)


original_data_path = r"C:\Users\18042\9444\raw_data"  
output_data_path = "data" 
train_ratio, val_ratio = 0.7, 0.2  

for split in ["train", "val", "test"]:
    split_path = os.path.join(output_data_path, split)
    os.makedirs(split_path, exist_ok=True)


class_to_images = {} 
for img_name, class_name in labels.items():
    class_to_images.setdefault(class_name, []).append(img_name)


for class_name, image_list in class_to_images.items():
    random.shuffle(image_list)
    num_train = int(len(image_list) * train_ratio)
    num_val = int(len(image_list) * val_ratio)

    split_data = {
        "train": image_list[:num_train],
        "val": image_list[num_train:num_train + num_val],
        "test": image_list[num_train + num_val:]
    }

    for split, images in split_data.items():
        class_dir = os.path.join(output_data_path, split, class_name)
        os.makedirs(class_dir, exist_ok=True)  

        for img_name in images:
            src_path = os.path.join(original_data_path, img_name)
            dst_path = os.path.join(class_dir, img_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"⚠️ worning: {src_path} not exists. Skipping...")

print("✅ Preprocessing done, added to 'data/' folder")


# In[ ]:




