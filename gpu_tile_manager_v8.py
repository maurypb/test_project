import torch
import numpy as np

class GPUTileManager:
    def __init__(self, source_images, target_images, tile_size, device='cuda', 
                 overlap_factor=1.5, use_grid_method=True):
        if len(source_images) != len(target_images):
            raise ValueError("Number of source and target images must be the same")
        
        self.source_images = source_images
        self.target_images = target_images
        self.tile_size = tile_size
        self.device = device
        self.overlap_factor = overlap_factor
        self.use_grid_method = use_grid_method

        # Initialize coverage tensors for each image
        self.coverage_tensors = [torch.zeros((img.shape[1], img.shape[2]), device=device) for img in source_images]
        self.current_tiles = []
        self.tile_index = 0

        # Calculate the total number of tiles to generate per epoch
        self.sum_of_image_areas = sum(img.shape[1] * img.shape[2] for img in source_images)
        self.tile_area = tile_size * tile_size
        self.total_tiles_per_epoch = int((self.sum_of_image_areas // self.tile_area) * self.overlap_factor)

        self.first_tile_of_epoch = True


    def generate_epoch_tiles(self):
        if self.use_grid_method:
            return self.generate_epoch_tiles_grid()
        else:
            return self._generate_epoch_tiles_original()

    def generate_epoch_tiles_grid(self):
        self.current_tiles = []
        
        for img_idx, img in enumerate(self.source_images):
            height, width = img.shape[1:]
            
            n_tiles_h, n_tiles_v, overlap_h, overlap_v, max_offset_h, max_offset_v = self._calculate_grid_parameters(width, height)
            
            for i in range(-1, n_tiles_v + 1):  # -1 to n_tiles_v+1 for extra top and bottom tiles
                for j in range(-1, n_tiles_h + 1):  # -1 to n_tiles_h+1 for extra left and right tiles
                    is_left_edge = (j <= 0)
                    is_right_edge = (j >= n_tiles_h - 1)
                    is_top_edge = (i <= 0)
                    is_bottom_edge = (i >= n_tiles_v - 1)
                    
                    base_y = int(i * (self.tile_size - overlap_v))
                    base_x = int(j * (self.tile_size - overlap_h))
                    
                    if is_left_edge or is_right_edge:
                        offset_x = 0
                    else:
                        offset_x = torch.randint(-int(max_offset_h), int(max_offset_h) + 1, (1,)).item()
                    
                    if is_top_edge or is_bottom_edge:
                        offset_y = 0
                    else:
                        offset_y = torch.randint(-int(max_offset_v), int(max_offset_v) + 1, (1,)).item()
                    
                    # Adjust base positions for edges
                    if is_left_edge:
                        base_x = 0
                    elif is_right_edge:
                        base_x = width - self.tile_size
                    
                    if is_top_edge:
                        base_y = 0
                    elif is_bottom_edge:
                        base_y = height - self.tile_size
                    
                    y = int(base_y + offset_y)
                    x = int(base_x + offset_x)
                    
                    # Ensure y and x are within bounds (should be unnecessary but kept for safety)
                    y = max(0, min(y, height - self.tile_size))
                    x = max(0, min(x, width - self.tile_size))
                    
                    self.current_tiles.append((img_idx, y, x))
                    self._update_coverage(img_idx, y, x)
        
        np.random.shuffle(self.current_tiles)
        self.tile_index = 0
        return len(self.current_tiles)



    def _calculate_grid_parameters(self, width, height):
        n_tiles_h = int(np.ceil((width - self.tile_size/2) / (self.tile_size/2))) + 1
        n_tiles_v = int(np.ceil((height - self.tile_size/2) / (self.tile_size/2))) + 1
        
        overlap_h = ((n_tiles_h * self.tile_size) - width) / (n_tiles_h - 1)
        overlap_v = ((n_tiles_v * self.tile_size) - height) / (n_tiles_v - 1)
        
        max_offset_h = min(overlap_h, self.tile_size / 4)
        max_offset_v = min(overlap_v, self.tile_size / 4)
        
        return n_tiles_h, n_tiles_v, overlap_h, overlap_v, max_offset_h, max_offset_v



    def _generate_epoch_tiles_original(self):
        """
        Generate tiles for an entire epoch, ensuring each image contributes proportionally to its size.
        
        This method handles both the initial random tile generation and subsequent least-covered area sampling.
        
        Returns:
            int: Total number of tiles generated for the epoch.
        """
        self.current_tiles = []
        
        for img_idx, coverage in enumerate(self.coverage_tensors):
            if self.first_tile_of_epoch:
                #self._generate_initial_tiles(img_idx)
                pass
            
            # Calculate the number of tiles to generate for this image based on its size
            image_size = self.source_images[img_idx].shape[1] * self.source_images[img_idx].shape[2]
            tiles_this_image = int((image_size / self.sum_of_image_areas) * self.total_tiles_per_epoch)
            tiles_this_image=1000
            # Generate tiles in batches for efficiency
            sample_batch_size = 10
            for _ in range((tiles_this_image // sample_batch_size) + 1):
                #new_tiles = self._sample_tiles_with_edge_bias(img_idx, tiles_this_image)
                #new_tiles = self._sample_tiles_uniform_pixel_representation(img_idx, tiles_this_image)
                new_tiles = self._sample_least_covered_areas(img_idx, min(sample_batch_size, tiles_this_image))
                self.current_tiles.extend([(img_idx, y, x) for y, x in new_tiles])
                for y, x in new_tiles:
                    self._update_coverage(img_idx, y, x)
                tiles_this_image -= len(new_tiles)
                if tiles_this_image <= 0:
                    break

        self.first_tile_of_epoch = False
        np.random.shuffle(self.current_tiles)
        self.tile_index = 0
        #self.reset_coverage()
        return len(self.current_tiles)



    def _update_coverage(self, img_idx, y, x):
        self.coverage_tensors[img_idx][y:y+self.tile_size, x:x+self.tile_size] += 1



    def _sample_least_covered_areas(self, img_idx, num_tiles):
        """
        Sample multiple least covered areas from the specified image.
        
        This method ensures uniform coverage over time by randomly selecting from the
        least covered areas, and then from the next least covered areas if necessary.
        
        Args:
            img_idx (int): Index of the image to sample from.
            num_tiles (int): Number of tiles to sample.
        
        Returns:
            list: List of (y, x) coordinates for the sampled tiles.
        """
        coverage = self.coverage_tensors[img_idx]
        height, width = coverage.shape
        # Use avg_pool2d to compute the average coverage for every possible tile position.
        # This operation slides a window of size tile_size x tile_size over the coverage tensor,
        # computing the average at each position. The result is a tensor where each element
        # represents the average coverage of a tile whose top-left corner is at that position.
        tile_coverage_sums = torch.nn.functional.avg_pool2d(
            coverage.unsqueeze(0).unsqueeze(0),
            kernel_size=self.tile_size,
            stride=1
        ).squeeze()

        #keeping this for diagnostics   
        #self.tile_coverage_sums[img_idx]=tile_coverage_sums

        # Find the minimum value and all indices with this value
        min_value = tile_coverage_sums.min()
        min_indices = torch.nonzero(tile_coverage_sums == min_value, as_tuple=False)
        num_min_indices = min_indices.size(0)

        if num_min_indices >= num_tiles:
            # If we have more minimum indices than needed, randomly select from them
            selected_indices = min_indices[torch.randperm(num_min_indices)[:num_tiles]]
        else:
            # Take all minimum indices
            selected_indices = min_indices
            
            # # If we need more, find the next least covered areas
            # if num_min_indices < num_tiles:
                #remaining_tiles = num_tiles - num_min_indices
                # # Mask out the minimum values we've already selected
                # mask = torch.ones_like(tile_coverage_sums, dtype=torch.bool)
                # mask[min_indices[:, 0], min_indices[:, 1]] = False
                # # Find the next least covered areas
                # next_least = torch.topk(tile_coverage_sums[mask], k=remaining_tiles, largest=False)
                # next_indices = torch.nonzero(mask, as_tuple=False)[next_least.indices]
                # # Combine the indices
                # selected_indices = torch.cat([selected_indices, next_indices])


        # tile_half_size = self.tile_size // 2
        # selected_indices = selected_indices - torch.tensor([tile_half_size, tile_half_size],device=self.device)
        # # Clamp the selected indices to ensure they're valid tile positions
        # y = torch.clamp(selected_indices[:, 0], 0, height - self.tile_size)
        # x = torch.clamp(selected_indices[:, 1], 0, width - self.tile_size)




        # Convert indices to y, x coordinates
        y, x = selected_indices[:, 0], selected_indices[:, 1]

        # Shuffle the selected coordinates
        perm = torch.randperm(len(y))
        y, x = y[perm], x[perm]

        return [(y[i].item(), x[i].item()) for i in range(len(y))]




    def _sample_tiles_uniform_pixel_representation(self, img_idx, num_tiles):
        """
        Sample tiles to ensure equal pixel representation and random tile distribution.

        Args:
            img_idx (int): Index of the image to sample from.
            num_tiles (int): Number of tiles to sample.

        Returns:
            list: List of (y, x) coordinates for the sampled tiles.
        """
        coverage = self.coverage_tensors[img_idx]
        height, width = coverage.shape

        # Total number of pixels
        total_pixels = height * width

        # Randomly sample pixel indices
        pixel_indices = torch.randint(0, total_pixels, (num_tiles,), device=self.device)
        pixel_y = pixel_indices // width
        pixel_x = pixel_indices % width

        # Calculate valid tile positions for each pixel
        tile_y_min = (pixel_y - self.tile_size + 1).clamp(0, height - self.tile_size)
        tile_y_max = pixel_y.clamp(0, height - self.tile_size)
        tile_x_min = (pixel_x - self.tile_size + 1).clamp(0, width - self.tile_size)
        tile_x_max = pixel_x.clamp(0, width - self.tile_size)

        # Compute the range of possible tile positions
        tile_y_range = tile_y_max - tile_y_min + 1  # +1 because upper bound is inclusive
        tile_x_range = tile_x_max - tile_x_min + 1

        # Generate random offsets within the range
        y_offsets = (torch.rand(num_tiles, device=self.device) * tile_y_range).floor().long()
        x_offsets = (torch.rand(num_tiles, device=self.device) * tile_x_range).floor().long()

        y = tile_y_min + y_offsets
        x = tile_x_min + x_offsets

        return [(y[i].item(), x[i].item()) for i in range(len(y))]






    def get_next_batch(self, batch_size):
        if self.tile_index + batch_size > len(self.current_tiles):
            self.generate_epoch_tiles()

        source_batch_tiles = []
        target_batch_tiles = []
        for _ in range(batch_size):
            img_idx, y, x = self.current_tiles[self.tile_index]
            source_tile = self.source_images[img_idx][:, y:y+self.tile_size, x:x+self.tile_size]
            target_tile = self.target_images[img_idx][:, y:y+self.tile_size, x:x+self.tile_size]
            source_batch_tiles.append(source_tile)
            target_batch_tiles.append(target_tile)
            self.tile_index += 1

        source_batch = torch.stack(source_batch_tiles).to(self.device, dtype=torch.float32)
        target_batch = torch.stack(target_batch_tiles).to(self.device, dtype=torch.float32)
        
        return source_batch, target_batch

    def reset_coverage(self):
        for c in self.coverage_tensors:
            c.zero_()

    def get_coverage_stats(self):
        return [(c.min().item(), (c == c.min()).sum().item(), c.max().item(), c.mean().item()) for c in self.coverage_tensors]