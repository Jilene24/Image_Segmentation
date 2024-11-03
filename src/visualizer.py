import torch
import matplotlib.pyplot as plt


class Visualizer:
    """Class for visualizing images and masks."""

    @staticmethod
    def show_image(image, mask):
        image = image.numpy().transpose(1, 2, 0)
        mask = mask.numpy().squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()
