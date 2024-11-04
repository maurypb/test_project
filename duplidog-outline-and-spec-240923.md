# Integrated duplidog MSRN Node Specification and Project Outline

Version 1.1 - September 23, 2024

## 1. Project Overview

### 1.1 Product Description
The duplidog MSRN Node is a machine learning tool designed to use a Multi-Scale Recurrent Network (MSRN) for image transformation tasks. It is developed as a standalone command-line tool, with potential future integration into ComfyUI and/or Autodesk Flame. The tool is targeted towards professional post-production and compositing tasks, such as rotoscoping, cleanup, and other repetitive tasks. 

### 1.1.1 Central Idea
- For professional post-production
- Works at high resolutions (HD 1920x1080 and UHD 3840x2160)
- Learns a transfer function from "before" to "after" based on a small set of representative frames
- Processes all frames through the learned transfer function
- Creates a shot-specific model for a particular problem, not a generalized model

### 1.2 Key Features
- MSRN-based learning of image transformations
- Efficient processing of large images using an advanced tiling mechanism
- Shot-specific training and inference
- Flexible loss functions including perceptual loss
- Model management (save, load, export)
- Robust augmentation system for both source and target images

### 1.3 Project Goals and Success Criteria
- Accurately learn and apply image transformations based on small training sets
- Efficiently process HD and UHD images using tiled processing
- Provide a smooth user experience in both standalone and future ComfyUI versions
- Demonstrate improved results compared to simpler image processing techniques
- Enable easy management of trained models
- Process a typical shot (100 frames at 2K resolution) in under 2 hours on target hardware

## 2. Key Decisions and Milestones

### 2.1 Training Set Generation Strategy #decision
- Implement a new strategy for generating training tiles:
  1. Create a large pool of tiles at the start of training:
     - Non-augmented tiles using weighted random sampling
     - Augmented tiles with various transformations
  2. For each epoch, randomly select a subset of these tiles
- This approach balances diversity and computational efficiency

### 2.2 Tile Sampling Method #decision
- Implement a weighted random sampling method for tile generation
- Use 1D weight distributions for height and width to favor edge regions
- Precompute weights for efficiency

### 2.3 Augmentation Strategy #decision
- Maintain a balance between non-augmented and augmented tiles
- Implement various augmentation techniques: rotation, scaling, color changes, etc.
- Allow for configurable ratios of augmented to non-augmented tiles

## 3. Current Challenges

### 3.1 Memory Management #challenge
- Balancing VRAM usage with processing speed for large datasets
- Efficiently storing and accessing the pre-generated tile pool

### 3.2 Computational Overhead #challenge
- Optimizing the initial tile pool generation process
- Ensuring fast tile selection for each epoch

### 3.3 Hyperparameter Tuning #challenge
- Determining optimal sizes for the tile pool and per-epoch subsets
- Balancing the ratio of augmented to non-augmented tiles

## 4. Next Steps

1. Implement the new tile generation and sampling strategy
2. Develop a memory-efficient storage system for the tile pool
3. Create a fast method for selecting tile subsets for each epoch
4. Implement and test the weighted random sampling for tile generation
5. Refine the augmentation techniques and make them configurable
6. Conduct experiments to determine optimal hyperparameters for the new approach

## 5. Recent Developments (Changelog)

- Proposed and discussed a new strategy for training set generation
- Developed a concept for weighted random sampling of tiles
- Explored memory management strategies for handling large datasets
- Considered a hybrid approach using both CPU and GPU memory for efficient processing

## 6. Current Conversation Summary

[See separate artifact for detailed conversation summary]

## 7. Project Glossary

- MSRN: Multi-Scale Recurrent Network
- Tile: A sub-section of an image used for training and processing
- Augmentation: The process of applying transformations to training data to increase diversity
- VRAM: Video Random Access Memory, used by GPUs for storing data
- Epoch: One complete pass through the training dataset

