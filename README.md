# visual-hull-mesh-creater

This is a code that create visual hull mesh data (.obj) from mask images and camera settings related to each image.<br>
The code will run on GPU because that is implemented with Pytorch.

<video src="https://github.com/user-attachments/assets/cd854a82-cb35-442a-bc57-4125a995ee56" controls="true" loop="true" autoplay="true" muted width="600"></video>
<!-- https://github.com/user-attachments/assets/cd854a82-cb35-442a-bc57-4125a995ee56 -->

## What you prepare

You have to prepare two types of data.

- Mask images like sample_data in `image_mask` dir.
- The json file that have camera settings like `transform_train.json` file.

The `transform_train.json` file have same format with NeRF-synthetic data.<br>
Please refer [NeRF-synthetic data format](https://www.matthewtancik.com/nerf "nerf project page").<br>
A few of images in `original_image_sample` dir are original images of mask images in `image_mask`.

Sample data are created from "www.cemyuksel.com/research/hairmodels".
