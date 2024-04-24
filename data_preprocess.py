import os
import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms


from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter


DATA_PATH = "/home/santo/Documents/Projects/My_Projects/Image Processing Computer Graphics Project/dataset/original-images"
OUTPUT_TRAIN_PATH = "train_data.h5"
OUTPUT_TEST_PATH = "test_data.h5"
random_crop = 30
patch_size = 32
label_size = 20
conv_side = 6
scale = 2
block_step = 16
block_size = 32

def save_patches(patches, folder):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    for i, patch in enumerate(patches):
        filename = f"{folder}/patch_{i}.png"
        patch_scaled = (patch * 255).astype(np.uint8)
        cv2.imwrite(filename, patch_scaled)

def prepare_train_data(image_path):

    hr_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb) #YCrCb separates the grayscale information from the color information allowing for more efficient compression and processing of image data.
    hr_img = hr_img[:, :, 0] #converts the image from YCrCb color space to grayscale by discarding color information
    shape = hr_img.shape
    # print(f"Shape of HR Image is: {image_path, shape}")

    """Downsampling by resizing to reduce the number of pixels in the image and 
        then upsampling the LR image to be of same size as HR original image"""
    
    lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale))) #shape[1] is the width of the original (HR) image, shape[0] is the height of the original image.
    lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

    """Calculate the number of non-overlapping blocks that can fit within the width and 
    height of the image for extracting patches from image"""

    width_num = int((shape[0] - (block_size - block_step) * 2) / block_step) #(block_size - block_step) * 2 calculates the total overlap introduced by the blocks in the height dimension
    height_num = int((shape[1] - (block_size - block_step) * 2) / block_step)

    data = []
    label = []
    # hr_patches = []
    # lr_patches = []

    for k in range(width_num):
        for j in range(height_num):
            x = k * block_step #Starting coordinates (x and y) of the current block
            y = j * block_step

            #Extract patches from both the HR image and the LR image
            hr_patch = hr_img[x: x + block_size, y: y + block_size]
            lr_patch = lr_img[x: x + block_size, y: y + block_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            #convert patch to pytorch tensor 
            lr = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).float()  #input image is grayscale so unsqueeze(0) is used to add batch dimension and channel dimen
            hr = torch.from_numpy(hr_patch[conv_side: -conv_side, conv_side: -conv_side]).unsqueeze(0).unsqueeze(0).float()
            # lists to hold the LR and HR patches for all blocks within the specific image
            data.append(lr)
            label.append(hr)
           
            #For saving the image pathces
            """
            hr_patches.append(hr_patch)
            lr_patches.append(lr_patch)

    hr_patches = np.array(hr_patches)
    lr_patches = np.array(lr_patches)
    save_patches(hr_patches, 'hr_patches_train')
    save_patches(lr_patches, 'lr_patches_train') """

    data = torch.stack(data)
    label = torch.stack(label)

    return data, label

def prepare_test_data(image_path):
    hr_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    shape = hr_img.shape

    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
    hr_img = hr_img[:, :, 0]

    lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
    lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

    """ Generate random coordinates (x and y) within the image boundaries to crop LR and HR image patches,
        ensure that different patches are sampled each time the function is called. So generate random patches 
        from different regions of the image """
    data = []
    label = []
    for _ in range(random_crop):
        x = np.random.randint(0, min(shape[0], shape[1]) - patch_size)
        y = np.random.randint(0, min(shape[0], shape[1]) - patch_size)

        lr_patch = lr_img[x: x + patch_size, y: y + patch_size]
        hr_patch = hr_img[x: x + patch_size, y: y + patch_size]

        lr_patch = lr_patch.astype(float) / 255.
        hr_patch = hr_patch.astype(float) / 255.

        lr = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).float()
        hr = torch.from_numpy(hr_patch[conv_side: -conv_side, conv_side: -conv_side]).unsqueeze(0).unsqueeze(0).float()

        data.append(lr)
        label.append(hr)

    data = torch.stack(data)
    label = torch.stack(label)

    return data, label


if __name__ == "__main__":
    # Load original images
    original_images = [os.path.join(DATA_PATH, img) for img in os.listdir(DATA_PATH)]

    # Split dataset into train and test sets
    train_images, test_images = train_test_split(original_images, test_size=0.2, random_state=42)

    # Prepare training data
    train_data = []
    train_label = []
    #count1 = 0
    for img_path in train_images:
        data, label = prepare_train_data(img_path)
        train_data.append(data)
        train_label.append(label)
        # count1 = count1 + 1 #For testing
        # if count1 == 1:
        #     break

    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)

    # Prepare test data
    test_data = []
    test_label = []
    # count2 = 0
    for img_path in test_images:
        data, label = prepare_test_data(img_path)
        test_data.append(data)
        test_label.append(label)
        # count2 = count2 + 1 #For testing 
        # if count2 == 5:
        #     break

    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)

    # Write to HDF5 files
    with h5py.File(OUTPUT_TRAIN_PATH, 'w') as h:
        h.create_dataset('data', data=train_data)
        h.create_dataset('label', data=train_label)
        print("Training data written successfully !")

    with h5py.File(OUTPUT_TEST_PATH, 'w') as h:
        h.create_dataset('data', data=test_data)
        h.create_dataset('label', data=test_label)
        print("Testing data written successfully !")

    # View contents of train_dataset
    print("Train Data Shape:", train_data.shape)
    print("Train Label Shape:", train_label.shape)

    print("Test Data Shape:", test_data.shape)
    print("Test Label Shape:", test_label.shape)
    
