import os
import cv2
import shutil
import numpy as np


class MIDV500DataProcessor:
    def __init__(self, dataset_dir, output_dir='midv500_data\\processed_midv500_data'):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

    @staticmethod
    def compute_skewness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        skewness = 0.0
        count = 0

        if lines is not None:
            for line in lines:
                for rho, angle in line:
                    skewness += angle
                    count += 1

        return skewness / count if count > 0 else 0.0

    @staticmethod
    def rotate_image(image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
        return rotated_image

    @staticmethod
    def enhance_and_denoise(image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to the grayscale channel
        enhanced_gray_image = cv2.equalizeHist(gray_image)
        # Merge the equalized channel with the original color channels
        enhanced_image = cv2.merge([enhanced_gray_image] * 3)
        # Apply denoising techniques (you can customize this part)
        denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

        return denoised_image

    def process_dataset(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        class_names = []

        # Traverse the dataset directory
        for class_name in os.listdir(self.dataset_dir):
            class_dir = os.path.join(self.dataset_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)

                if not os.path.exists(os.path.join(self.output_dir, 'midv500', class_name)):
                    print(f"Processing class: {class_name}")
                    # Create output directories for the current class
                    output_class_dir = os.path.join(self.output_dir, 'midv500', class_name)
                    os.makedirs(output_class_dir, exist_ok=True)
                    os.makedirs(os.path.join(output_class_dir, 'ground_truth'), exist_ok=True)
                    os.makedirs(os.path.join(output_class_dir, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(output_class_dir, 'videos'), exist_ok=True)

                    # Iterate through images in the current class
                    for sub_dir_name in os.listdir(os.path.join(class_dir, 'images')):
                        sub_dir_path = os.path.join(class_dir, 'images', sub_dir_name)
                        if os.path.isdir(sub_dir_path):
                            print(f"  Processing sub-directory: {sub_dir_name}")
                            output_sub_dir_path = os.path.join(output_class_dir, 'images', sub_dir_name)
                            os.makedirs(output_sub_dir_path, exist_ok=True)
                            for file_name in os.listdir(sub_dir_path):
                                image_path = os.path.join(sub_dir_path, file_name)
                                output_image_path = os.path.join(output_sub_dir_path, file_name)
                                print(f"    Processing image: {file_name}")

                                # Check if the processed file already exists
                                if os.path.exists(output_image_path):
                                    print(f"    Image already processed: {file_name}")
                                    continue

                                image = cv2.imread(image_path)

                                # Rotation Correction
                                if self.requires_rotation_correction(image):
                                    rotated_image = self.rotate_image(image, -self.compute_skewness(image))
                                    # Image Enhancement and de-noising
                                    processed_image = self.enhance_and_denoise(rotated_image)
                                    processed_output_path = os.path.join(output_sub_dir_path, file_name)
                                    cv2.imwrite(processed_output_path, processed_image)
                                else:
                                    output_image_path = os.path.join(output_class_dir, 'images', sub_dir_name,
                                                                     file_name)
                                    cv2.imwrite(output_image_path, image)
                        else:
                            print(f"  Copying non-directory file: {sub_dir_path}")
                            output_sub_dir_path = os.path.join(output_class_dir, 'images')
                            shutil.copy(sub_dir_path, output_sub_dir_path)

                    # Copy ground truth files
                    for gt_dir_name in os.listdir(os.path.join(class_dir, 'ground_truth')):
                        gt_dir_path = os.path.join(class_dir, 'ground_truth', gt_dir_name)
                        if os.path.isdir(gt_dir_path):
                            print(f"  Processing ground truth directory: {gt_dir_name}")
                            output_gt_dir_path = os.path.join(output_class_dir, 'ground_truth', gt_dir_name)
                            os.makedirs(output_gt_dir_path, exist_ok=True)

                            for file_name in os.listdir(gt_dir_path):
                                gt_file_path = os.path.join(gt_dir_path, file_name)
                                output_gt_file_path = os.path.join(output_gt_dir_path, file_name)
                                # Check if the processed file already exists
                                if os.path.exists(output_gt_file_path):
                                    print(f"    Ground truth file already processed: {file_name}")
                                    continue
                                shutil.copy(gt_file_path, output_gt_file_path)
                        else:
                            print(f"  Copying non-directory ground truth file: {gt_dir_path}")
                            output_dir_path = os.path.join(output_class_dir, 'ground_truth', gt_dir_name)
                            # Check if the processed file already exists
                            if not os.path.exists(output_dir_path):
                                shutil.copy(gt_dir_path, output_dir_path)

                    # Copy video files
                    for file_name in os.listdir(os.path.join(class_dir, 'videos')):
                        video_file_path = os.path.join(class_dir, 'videos', file_name)
                        output_video_path = os.path.join(output_class_dir, 'videos', file_name)
                        print(f"  Copying video file: {file_name}")

                        # Check if the processed file already exists
                        if not os.path.exists(output_video_path):
                            shutil.copy(video_file_path, output_video_path)
                else:
                    print(f"files of {class_name} are already processed and stored in the output directory")

        total_classes = len(class_names)
        return total_classes, class_names

    def requires_rotation_correction(self, image, skewness_threshold=0.5):
        skewness = self.compute_skewness(image)
        return abs(skewness) > skewness_threshold
