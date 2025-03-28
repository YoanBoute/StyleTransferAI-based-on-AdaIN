from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch

os.environ["KERAS_BACKEND"] = "torch"
import keras
from pathlib import Path


class AdaINDataset(Dataset) :
    def __init__(self, files_list, transform_pipeline) :
        super().__init__()
        self.files_list = files_list
        self.transform_pipeline = transform_pipeline

    def __len__(self) :
        return len(self.files_list)
    
    def __getitem__(self, index):
        content_file, style_file = self.files_list[index]

        # Loading the images only when required prevents from having to load the whole dataset at once, which would break the system
        content_img = keras.utils.load_img(content_file)
        style_img = keras.utils.load_img(style_file)
        
        content_tensor = self.transform_pipeline(content_img)
        style_tensor = self.transform_pipeline(style_img)
        
        return (content_tensor, style_tensor), torch.tensor(0.0) # The null tensor is returned as a fake label data, whose existence is needed by Keras


class SequentialRepetingSampler(Sampler) :
    def __init__(self, dataset_size, data_source = None):
        super().__init__(data_source)
        self.dataset_size = dataset_size
        self.indices = np.arange(self.dataset_size)
        self.pos = -1
    
    def __iter__(self):
        while True :
            self.pos += 1
            if self.pos >= self.dataset_size :
                np.random.shuffle(self.indices)
                self.pos = 0
            # print(self.pos, '/', self.dataset_size)
            yield self.indices[self.pos]
            
    
    def __len__(self) :
        return self.dataset_size


class AdaINDataModule() :
    def __init__(
            self, 
            content_img_dir, 
            style_img_dir, *,
            iterate_subdirs = True, 
            imgs_size = 512, 
            crop_window_size = (256,256), 
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

        self.imgs_size = imgs_size
        self.crop_window_size = crop_window_size

        self.dataset_size = dataset_size
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.seed = seed if seed is not None else np.random.randint(1e6)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def create_datasets(self) :
        if self._are_datasets_created :
            print("Datasets are already created")
            return

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
        
        train_set_size, val_set_size, test_set_size = (np.array(self.train_val_test_split) * self.dataset_size).round().astype(int).tolist()
        train_val_indices, test_indices = train_test_split(np.arange(self.dataset_size), test_size=test_set_size)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=val_set_size)

        content_files = np.array(content_files)
        style_files = np.array(style_files)
        train_images = np.column_stack((content_files[train_indices], style_files[train_indices]))
        val_images = np.column_stack((content_files[val_indices], style_files[val_indices]))
        test_images = np.column_stack((content_files[test_indices], style_files[test_indices]))

        transform_pipeline = transforms.Compose((
            transforms.Resize(self.imgs_size),
            transforms.RandomCrop(self.crop_window_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.permute(1, 2, 0)),
            transforms.Lambda(lambda x : 255 * x), # Scale images to [0, 255] as expected for VGG19
        ))

        self._train_dataset = AdaINDataset(train_images, transform_pipeline)
        self._val_dataset = AdaINDataset(val_images, transform_pipeline)
        self._test_dataset = AdaINDataset(test_images, transform_pipeline)
        
        self._are_datasets_created = True    

    def train_dataloader(self) :
        if not self._are_datasets_created :
            self.create_datasets()
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=False, sampler=SequentialRepetingSampler(len(self._train_dataset))) # No need to shuffle here as the datasets are already shuffled (if self.shuffle is enabled)

    def val_dataloader(self) :
        if not self._are_datasets_created :
            self.create_datasets()
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, sampler=SequentialRepetingSampler(len(self._val_dataset))) # No need to shuffle here as the datasets are already shuffled (if self.shuffle is enabled)

    def test_dataloader(self) :
        if not self._are_datasets_created :
            self.create_datasets()
        return DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False, sampler=SequentialRepetingSampler(len(self._test_dataset))) # No need to shuffle here as the datasets are already shuffled (if self.shuffle is enabled)


    def __setattr__(self, name, value):
        match name :
            case '_are_datasets_created' :
                super().__setattr__('_are_datasets_created', value)
            case _ :
                if self.__dict__.get("_are_datasets_created") is None or self._are_datasets_created :
                    raise PermissionError("You don't have the permission to change attributes once the datasets are created. Please create a new DataModule")
                else : 
                    super().__setattr__(name, value)