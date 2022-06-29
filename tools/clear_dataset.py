import json
import random
from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
import torchvision.models as models
import mmcls.models as mmcls_models
import models as my_models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)
        return sample, label


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    with open(file_path, 'w+') as f:
        json.dump(data, f)


def get_fewshot_indices(
        folder_path: str,
        num_classes: int = 11,
        bucket_0: bool = False,
        shot: int = 5,
        seed: int = 0):
    """Generate a few-shot train-test split.

    Args:
        folder_path (str): Path to the folder containing the CLEAR dataset.
        num_classes (int, optional): Number of classes. Defaults to 11.
        bucket_0 (bool, optional): Whether to include the 0th bucket. Defaults to False.
        shot (int, optional): Number of examples per class in the training set. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 0.
    """
    folder_path = Path(folder_path)
    # bucket_indices
    bucket_indices = [str(i + 1) for i in range(10)]
    if bucket_0:
        bucket_indices.insert(0, '0')
    # get few-shot file list
    for b_idx in bucket_indices:
        print(f'Processing bucket {b_idx}')
        all_images_b_idx = []
        class_indices = {i: [] for i in range(num_classes)}
        with open(folder_path / 'training_folder/filelists' / b_idx / 'all.txt') as f:
            for line in f:
                img, class_index = line.strip().split(' ')
                class_index = int(class_index)
                all_images_b_idx.append((class_index, img))
                class_indices[class_index].append(len(all_images_b_idx) - 1)
        # get few-shot indices
        split_name = f'{shot}_shot'
        seed_name = f'split_{seed}'
        split_folder_path = folder_path / 'training_folder' / split_name / seed_name / b_idx
        split_folder_path.mkdir(exist_ok=True, parents=True)
        train_indices = []
        test_indices = []
        for c, indices in class_indices.items():
            random.seed(seed)
            random.shuffle(indices)
            train_indices.extend(indices[:shot])
            test_indices.extend(indices[shot:])
        # save few-shot indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        train_indices_path = split_folder_path / 'train_indices.json'
        test_indices_path = split_folder_path / 'test_indices.json'
        save_json(train_indices, train_indices_path)
        save_json(test_indices, test_indices_path)
        with open(split_folder_path / 'train.txt', 'w+') as f:
            for i in train_indices:
                f.write(f'{all_images_b_idx[i][1]} {all_images_b_idx[i][0]}\n')
        with open(split_folder_path / 'test.txt', 'w+') as f:
            for i in test_indices:
                f.write(f'{all_images_b_idx[i][1]} {all_images_b_idx[i][0]}\n')


def get_pretrained_features(
        folder_path: str,
        feature_name: str,
        checkpoint_path: str = None,
        arch: Union[str, dict] = 'resnet50',
        bucket_0: bool = False):
    """Compute and save features of pretrained model for CLEAR.
    Check https://github.com/linzhiqiu/continual-learning for more details.

    Args:
        folder_path (str): Path to the folder containing the CLEAR dataset.
        feature_name (str): Name of the feature.
        checkpoint_path (str, optional): Path to the checkpoint.
        arch (str | dict, optional): Model architecture. Defaults to 'resnet50'.
        bucket_0 (bool, optional): Whether to include the 0th bucket. Defaults to False.
    """
    # build model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if isinstance(arch, str):
        model = getattr(models, arch)(pretrained=checkpoint_path is None)
    elif isinstance(arch, dict):
        model_type = arch.pop('type')
        try:
            model = getattr(mmcls_models, model_type)(**arch)
        except AttributeError:
            model = getattr(my_models, model_type)(**arch)
    else:
        raise NotImplementedError
    model.fc = torch.nn.Identity()
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if len(state_dict) == 1:
            state_dict = state_dict[list(state_dict)[0]]
        model.load_state_dict(state_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to('cuda')
    # get features
    folder_path = Path(folder_path)
    feature_path = folder_path / 'training_folder/features' / feature_name
    bucket_indices = [str(i + 1) for i in range(10)]
    if bucket_0:
        bucket_indices.insert(0, '0')
    for b_idx in bucket_indices:
        feature_path_b_idx = feature_path / b_idx
        feature_path_b_idx.mkdir(parents=True, exist_ok=True)
        all_images_b_idx = []
        with open(folder_path / 'training_folder/filelists' / b_idx / 'all.txt') as f:
            for line in f:
                img, class_id = line.strip().split(' ')
                class_id = int(class_id)
                all_images_b_idx.append((folder_path / img, class_id))
        loader = DataLoader(
            SimpleDataset(all_images_b_idx, test_transform),
            batch_size=512)
        feature_list = []
        label_list = []
        with torch.no_grad():
            for img, label in tqdm(loader, desc=f'Bucket {b_idx}'):
                img = img.cuda()
                feature = model(img)
                if isinstance(feature, tuple):
                    feature = feature[-1]
                if isinstance(feature, list):
                    feature = feature[-1]
                feature_list.append(feature.cpu())
                label_list.extend(label.numpy())
        features = torch.cat(feature_list, dim=0)
        torch.save((features, label_list), feature_path_b_idx / 'all.pth')
    print('Saving to', feature_path)
