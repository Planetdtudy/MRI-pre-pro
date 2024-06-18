#dicoms to pngs

import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt

#read DICOM 
def read_dicom_image(file_path):
    dicom = pydicom.dcmread(file_path)
    img = dicom.pixel_array.astype(np.uint16)
    return img


input_file = 'path'
image_16bit = read_dicom_image(input_file)
# Assuming 'image_16bit' is your 16-bit input array
# For the sake of demonstration, let's create a synthetic 16-bit image
#image_16bit = np.random.randint(0, 65536, (512, 512), dtype=np.uint16)

# Define the breakpoints for the three parts
breakpoints = [0, 21845, 43690, 65535]

# Function to scale a part of the image to 8-bit
def scale_to_8bit(image_part, min_val, max_val):
    # Scale values to 0-255 range
    scaled = 255 * (image_part - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 255).astype(np.uint8)

# Create the three 8-bit parts
red_channel = scale_to_8bit(np.clip(image_16bit, breakpoints[0], breakpoints[1]), breakpoints[0], breakpoints[1])
green_channel = scale_to_8bit(np.clip(image_16bit, breakpoints[1], breakpoints[2]), breakpoints[1], breakpoints[2])
blue_channel = scale_to_8bit(np.clip(image_16bit, breakpoints[2], breakpoints[3]), breakpoints[2], breakpoints[3])

# Combine into an RGB image
rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

# Plot the channels and the combined RGB image
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(red_channel, cmap='Reds')
ax[0].set_title('Red Channel')
ax[0].axis('off')

ax[1].imshow(green_channel, cmap='Greens')
ax[1].set_title('Green Channel')
ax[1].axis('off')

ax[2].imshow(blue_channel, cmap='Blues')
ax[2].set_title('Blue Channel')
ax[2].axis('off')

ax[3].imshow(rgb_image)
ax[3].set_title('RGB Image')
ax[3].axis('off')

plt.show()



# convert 16-bit image to 8-bit image with contrast preservation
def convert_16bit_to_8bit(img):
    #  histogram equalization
    hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])  # Collect 16 bits histogram # Flattening is a technique that is used to convert multi-dimensional arrays into a 1-D array
    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.plot(bins[:-1], hist)  # bins[:-1] because np.histogram returns bins with one extra element
    plt.title('Histogram')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.show()

    n_bins = 20
    dist=img.flatten()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(dist, bins=n_bins)
    plt.show()


    cdf = hist.cumsum()
    cdf_m = cdf#np.ma.masked_equal(cdf, 0)  # Mask zeros
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint16')
    
    
    img_equalized = cdf[img]

    # Normalize 
    img_8bit = cv2.normalize(img_equalized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return img_8bit

# plot before and after images
def plot_images(before, after):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original 16-bit Image')
    plt.imshow(before, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Converted 8-bit Image')
    plt.imshow(after, cmap='gray')
    plt.axis('off')
    
    plt.show()

# process the DICOM image
def process_dicom_image(input_file, output_file):
    img_16bit = read_dicom_image(input_file)
    img_8bit = convert_16bit_to_8bit(img_16bit)

    # Save PNG
    cv2.imwrite(output_file, img_8bit)

    # Plot
    plot_images(img_16bit, img_8bit)


output_directory = 'path'
dicom_folder = 'path'
input_file = 'path'
output_file = os.path.join(output_directory, os.path.basename(dicom_folder) + '.png')

process_dicom_image(input_file, output_file)
