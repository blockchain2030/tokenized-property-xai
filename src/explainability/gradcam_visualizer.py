"""Grad-CAM placeholder interface. Use with ResNet-50 layer4 feature maps."""
import torch
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def generate(self, image_tensor, class_score):
        self.model.zero_grad()
        class_score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.squeeze().cpu().numpy()

def save_heatmap_overlay(original_image, heatmap, output_path):
    plt.imshow(original_image)
    plt.imshow(heatmap, alpha=0.45)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
