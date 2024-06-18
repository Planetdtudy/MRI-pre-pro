# MRI-pre-pro
Pre-processing MRI images for further deep learning.

This project is about MRI images pre-processing for further deep-learning usage.
There doesn’t exist any standard preprocessing procedures for MRI images. Usually you get MRI data in .nii or .dicom formats, while major deep learning models are designed to be fed with .png files.
Here we face one of the major issues: loosing information.
Some scientist convert data into numpy array, which increases the speed of download, but eventually you still loosing going from 16-bit to 8 -bit data. 
Another issue is that we might to crop the images for a size the DL model was trained on. Sometimes it can be 224 * 224 or 299 * 299, when primary you have 512 * 512.

The other question is the normalization. 
For regular images (nature images) the min-max normalization is widely used.
For MRI images intensity normalization differs from natural images. For more information, why it is so visit https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1, where it is nicely described.

So, this package propose some methods to be used for MRI images.
