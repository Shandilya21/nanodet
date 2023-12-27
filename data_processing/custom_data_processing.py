import os
import midv500
from data_transformation_rules import MIDV500DataProcessor


def get_data(data_dir, dataset_name):
    midv500.download_dataset(data_dir, dataset_name)


def convert_to_coco(data_dir, filename, export_dir):
    # convert annotations to coco format
    midv500.convert_to_coco(data_dir, export_dir, filename)


def check_dataset_dir(data_dir):
    return os.path.exists(data_dir)


def apply_data_processing(dataset_directory):
    processor = MIDV500DataProcessor(dataset_directory)
    total_class, class_names = processor.process_dataset()
    return total_class, class_names


if __name__ == "__main__":
    dataset_dir = 'midv500_data/'
    dataset_name = "midv500"
    filename = 'midv500_processed'
    export_dir = 'data_processing'

    if not check_dataset_dir(dataset_dir):
        get_data(dataset_dir, dataset_name)
    else:
        print(f"The dataset directory '{dataset_dir}' is already present. Skipping download.")
    data_directory = 'midv500_data/midv500/'
    total_classes, class_names = apply_data_processing(data_directory)
    print(f"Total Classes: {total_classes}")
    print(f"Class Names: {class_names}")

    convert_to_coco(dataset_dir, filename, export_dir)
