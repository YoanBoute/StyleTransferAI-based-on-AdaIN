import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms

os.environ["KERAS_BACKEND"] = "torch"
from pathlib import Path


class AdaINDataModule() :
    def __init__(
            self, 
            content_img_dir, 
            style_img_dir, *,
            iterate_subdirs = True, 
            imgs_size = 512, 
            keep_aspect_ratio = True, 
            crop_window_size = (256,256), 
            # num_crops_per_img = 3,
            dataset_size = 80000, 
            train_val_test_split = (0.8, 0.1, 0.1),
            batch_size = 8,
            shuffle = True,
            seed = None
        ) :
        self._are_datasets_created = False

        self.content_img_dir = content_img_dir
        self.style_img_dir = style_img_dir
        self.iterate_subdirs = iterate_subdirs

        self.keep_aspect_ratio = keep_aspect_ratio
        self.imgs_size = imgs_size
        self.crop_window_size = crop_window_size
        # self.num_crops_per_img = num_crops_per_img

        self.dataset_size = dataset_size
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.seed = seed if seed is not None else np.random.randint(1e6)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def create_datasets(self) :
        if self.are_datasets_created :
            answer = input("Warning : Datasets have already been created in this Data Module. Restarting the creation will erase these datasets. Proceed anyway ? [y/n]")
            if answer[0] not in ["y", "Y"] :
                return
            else :
                self._train_dataset = None
                self._val_dataset = None
                self._test_dataset = None

        if self.iterate_subdirs :
            content_files = [f for f in self.content_img_dir.glob("**/*.jpg")]
            style_files = [f for f in self.style_img_dir.glob("**/*.jpg")]
        else :
            content_files = [f for f in self.content_img_dir.glob("*.jpg")]
            style_files = [f for f in self.style_img_dir.glob("*.jpg")]
        
        len_content = len(content_files)
        len_style = len(style_files)

        # If the file list of either content or style images is shorter than the expected dataset size, some images are randomly duplicated to reach this size
        # This will in the vast majority of cases lead to new content_style pairs, even if content or syle is a duplicate 
        if len_content < self.dataset_size :
            content_files.extend(np.random.choice(content_files, size=self.dataset_size - len_content, replace=True).tolist())
        if len_style < self.dataset_size :
            style_files.extend(np.random.choice(style_files, size=self.dataset_size - len_style, replace=True).tolist())

        if self.shuffle :
            np.random.seed(self.seed) # Setting the seed here ensures a reproducible shuffling
            np.random.shuffle(content_files)
            np.random.shuffle(style_files)

        if len_content > self.dataset_size :
            content_files = content_files[:self.dataset_size]
        if len_style > self.dataset_size :
            style_files = style_files[:self.dataset_size]
        
        train_set_size, val_set_size, test_set_size = (np.array(self.train_val_test_split) * self.dataset_size).round().astype(int)
        train_val_indices, test_indices = train_test_split(np.arange(self.dataset_size), test_set_size)
        train_indices, val_indices = train_test_split(train_val_indices, val_set_size)

        content_files = np.array(content_files)
        style_files = np.array(style_files)
        train_images = zip(content_files[train_indices], style_files[train_indices])
        val_images = zip(content_files[val_indices], style_files[val_indices])
        test_images = zip(content_files[test_indices], style_files[test_indices])

        transform_pipeline = transforms.Compose(
            transforms.Resize(self.imgs_size),
            transforms.RandomCrop(self.crop_window_size),
            transforms.ToTensor()
        )

        self._train_dataset = AdaINDataset(train_images, transform_pipeline)
        self._val_dataset = AdaINDataset(val_images, transform_pipeline)
        self._test_dataset = AdaINDataset(test_images, transform_pipeline)
        
        self._are_datasets_created = True
        
    
    def train_dataset(self) :
        if self._train_dataset is None :
            self.create_datasets()
        return self._train_dataset
    
    def val_dataset(self) :
        if self._val_dataset is None :
            self.create_datasets()
        return self._val_dataset
    
    def test_dataset(self) :
        if self._test_dataset is None :
            self.create_datasets()
        return self._test_dataset
    

    def train_dataloader(self) :
        pass

    def val_dataloader(self) :
        pass

    def test_dataloader(self) :
        pass
    

    @property
    def are_datasets_created(self) :
        return self._are_datasets_created
    
    @are_datasets_created.setter
    def are_datasets_created(self, new_value) :
        raise AttributeError("This is a private attribute that should not be modified outside of the class")
