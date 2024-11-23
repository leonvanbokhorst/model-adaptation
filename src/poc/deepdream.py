import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class DeepDreamer:
    def __init__(self, model_name="inception_v3", layer_name="Mixed_5b"):
        # Update model initialization to use weights parameter
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights)
        self.model.eval()

        # Dictionary to store activations
        self.activations = {}
        self.layer_name = layer_name

        # Register forward hook
        for name, layer in self.model.named_modules():
            if name == layer_name:
                layer.register_forward_hook(self._get_activation(name))

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output

        return hook

    def preprocess_image(self, image_path, size=512):
        image = Image.open(image_path)
        # Resize while maintaining aspect ratio
        ratio = size / min(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)
        image = image.resize(new_size, Image.LANCZOS)

        # Convert to tensor and normalize
        loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = loader(image).unsqueeze(0)
        return image

    def deprocess_image(self, tensor):
        # Convert back to image
        tensor = tensor.squeeze(0)
        # Denormalize
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(
            3, 1, 1
        ) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to PIL image
        transform = transforms.ToPILImage()
        return transform(tensor)

    def dream(
        self, image_path, num_iterations=20, lr=0.01, octave_scale=1.4, num_octaves=4
    ):
        # Load base image
        base_img = self.preprocess_image(image_path)

        # Create octaves pyramid
        octaves = []
        for _ in range(num_octaves - 1):
            octaves.append(base_img)
            base_img = torch.nn.functional.interpolate(
                base_img,
                scale_factor=1 / octave_scale,
                mode="bicubic",
                align_corners=False,
            )

        detail = None
        for octave_idx, octave_base in enumerate(reversed(octaves)):
            if detail is not None:
                detail = torch.nn.functional.interpolate(
                    detail,
                    size=octave_base.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Add detail from previous octave to current
            input_img = octave_base + detail if detail is not None else octave_base
            input_img = input_img.detach()  # Detach from previous graph
            input_img.requires_grad_(True)  # Enable gradients for new iteration

            for i in range(num_iterations):
                # Clear gradients at start of iteration
                if input_img.grad is not None:
                    input_img.grad.zero_()

                # Forward pass
                out = self.model(input_img)
                activation = self.activations[self.layer_name]

                # Calculate loss
                loss = activation.norm()  # Remove negative sign for maximization

                # Backward pass
                loss.backward()

                # Ensure we have gradients
                if input_img.grad is not None:
                    # Gradient normalization and update
                    grad = input_img.grad.data
                    grad_mean = grad.abs().mean()
                    grad_norm = grad / (grad_mean + 1e-8)
                    input_img.data += lr * grad_norm

                    # Apply image regularization
                    input_img.data = torch.clamp(input_img.data, -1, 1)

                if (i + 1) % 5 == 0:
                    print(
                        f"Octave {octave_idx+1}/{num_octaves}, "
                        f"Iteration {i+1}/{num_iterations}, "
                        f"Loss: {loss.item():.3f}"
                    )

            # Extract detail produced in this octave
            detail = input_img.data - octave_base

        return self.deprocess_image(input_img.detach())


# Example usage
def generate_dream(image_path, output_path, iterations=7, lr=0.09):
    dreamer = DeepDreamer()
    dreamed_image = dreamer.dream(
        image_path, num_iterations=iterations, lr=lr, octave_scale=1.9, num_octaves=4
    )
    dreamed_image.save(output_path)

    # Display original and dreamed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(Image.open(image_path))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(dreamed_image)
    ax2.set_title("DeepDream Image")
    ax2.axis("off")

    plt.show()


if __name__ == "__main__":
    generate_dream("images/cat.jpg", "images/dreamed_cat.jpg")
