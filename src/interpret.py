from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

def generate_heatmap(model, image, target_class):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
    grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(target_class)])[0]
    rgb_img = image.squeeze().permute(1,2,0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization
