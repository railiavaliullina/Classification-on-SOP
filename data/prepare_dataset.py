""" module for getting train and valid set paths, which wil be used while getting Dataset """
import pickle
from os import walk

from configs.config import cfg


def get_paths(cfg, dataset_type, save_to_pickle=True):
    dataset_path = cfg['data']['dataset_path'] + dataset_type + "/"  # dir with images
    _, classes_dirs, _ = next(walk(dataset_path))
    paths, labels = [], []
    for class_id, class_dirs in enumerate(classes_dirs):
        dir, _, class_filenames = next(walk(dataset_path + class_dirs))
        class_paths = [dir + '/' + f for f in class_filenames]
        paths.extend(class_paths)
        labels.extend([class_id] * len(class_paths))
    if save_to_pickle:
        with open(cfg['data']['images_paths_file_path'] + dataset_type + '_paths_and_labels.pickle', 'wb') as f:
            pickle.dump({'paths': paths, 'labels': labels}, f)
    return paths


if __name__ == '__main__':
    train_paths = get_paths(cfg, "train")
    valid_paths = get_paths(cfg, "valid")
