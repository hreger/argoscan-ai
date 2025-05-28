import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output)

        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot_output)

        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[:, i].view(-1, 1, 1)

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()

def apply_heatmap(image_path, heatmap, save_path=None, alpha=0.5):
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Resize heatmap to match image dimensions
    heatmap = Image.fromarray(np.uint8(255 * heatmap))
    heatmap = heatmap.resize((img_array.shape[1], img_array.shape[0]))
    heatmap = np.array(heatmap)

    # Apply colormap
    colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Blend original image with heatmap
    overlayed_img = np.uint8(img_array * (1 - alpha) + colored_heatmap * alpha)

    if save_path:
        Image.fromarray(overlayed_img).save(save_path)

    return overlayed_img