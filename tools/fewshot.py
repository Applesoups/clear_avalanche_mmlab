import json
import random
from pathlib import Path


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
        num_buckets: int = 10,
        shot: int = 5,
        seed: int = 0):
    """Generate a few-shot train-test split.

    Args:
        folder_path (str): Path to the folder containing the CLEAR dataset.
        num_classes (int, optional): Number of classes. Defaults to 11.
        num_buckets (int, optional): Number of buckets. Defaults to 10.
        shot (int, optional): Number of examples per class in the training set. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 0.
    """
    folder_path = Path(folder_path)
    # only consider buckets 1-10
    bucket_indices = [str(i + 1) for i in range(num_buckets)]
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
