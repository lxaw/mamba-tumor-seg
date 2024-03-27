##################################
# Clay Crews, Lex Whalen
#
# Dataset class for Brain MRI Transforms.
# Only reshapes to 256x256.
# For use in later pipelines.

from torchvision import transforms

class BrainMRITransforms:
    def __init__(self, resize_shape=(256, 256)):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize_shape)
        ])

    def __call__(self, img):
        return self.transforms(img)