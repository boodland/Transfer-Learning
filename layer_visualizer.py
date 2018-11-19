from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
#import torch.nn as nn
import cv2
import os

class LayerVisualizer():
    def __init__(self, layers_interceptor):
        self.layers_interceptor = layers_interceptor
        self.__process_outputs()
        
    def __process_outputs(self):
        self.mean_backward_outputs = []
        self.forward_outputs = []
        for layer_interceptor in self.layers_interceptor:
            forward_outputs = layer_interceptor.forward_outputs
            self.forward_outputs.append(forward_outputs)
            
            mean_backward_outputs = np.mean(layer_interceptor.backward_outputs, axis=(0,2,3))
            self.mean_backward_outputs.append(mean_backward_outputs)
            
    def __plot_channels(self, channels, title):
        images_per_row = 16
        n_features = channels.shape[0]
        size = channels.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = channels[
                    col * images_per_row + row
                    :, :
                ]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image[0]

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(title, fontsize=20)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.show()
        
    def display_layers_channels_output(self, num_layers=None):
        num_layers = num_layers if num_layers else len(self.layers_interceptor)
        for layer_num in range(num_layers):
            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the gradient
            forward_output = self.forward_outputs[layer_num]
            self.__plot_channels(forward_output.copy(), f'Channels Output Of Selected Layer {layer_num+1} With Respect The Input')
            
            forward_output_copy = forward_output.copy()
            for i in range(forward_output_copy.shape[0]):
                forward_output_copy[i, :, :] *= self.mean_backward_outputs[layer_num][i]

            self.__plot_channels(forward_output_copy, f'Channels Output Of Selected Layer {layer_num+1} With Respect The Label')
            
    def __get_layers_heatmap(self):
        heatmaps = []
        for forward_outputs in self.forward_outputs:
            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
            heatmap = np.mean(forward_outputs, axis=0)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmaps.append(heatmap)
        return heatmaps
            
    def __generate_image_activation(self, heatmaps, img_filename, generated_image_filename):
        for ind, heatmap in enumerate(heatmaps, 1):
            # We use cv2 to load the original image
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (224, 224))

            # We resize the heatmap to have the same size as the original image
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            # We convert the heatmap to RGB
            heatmap = np.uint8(255 * heatmap)

            # We apply the heatmap to the original image
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 0.4 here is a heatmap intensity factor
            superimposed_img = heatmap * 0.4 + img

            # Save the image to disk
            cv2.imwrite(generated_image_filename.format(ind), superimposed_img)

    def __display_images_activation(self, generated_image_filename):
        heatmaps = []
        num_layers = len(self.layers_interceptor)
        fig, axs = plt.subplots(2, num_layers, figsize=(60, 20))
        fig.suptitle('Image Activation With Respect The Label', fontsize=60)
        flat_axs = axs.reshape(-1)
        for forward_outputs, ax in zip(self.forward_outputs, flat_axs[:num_layers]):
            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
            heatmap = np.mean(forward_outputs, axis=0)

            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            ax.matshow(heatmap)
            ax.axis('off')

            heatmaps.append(heatmap)
        for ind, ax in enumerate(flat_axs[num_layers:], 1): 
            img = Image.open(generated_image_filename.format(ind))
            ax.imshow(img)
            ax.axis('off')
        plt.show()
    
    def display_layers_activation(self, img_filename):
        img_path = os.path.dirname(img_filename)
        generated_image_filename = img_path+'/selected_layer_{}.jpg'
        heatmaps = self.__get_layers_heatmap()
        self.__generate_image_activation(heatmaps, img_filename, generated_image_filename)
        self.__display_images_activation(generated_image_filename)
    