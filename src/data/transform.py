from torchvision import transforms

class TransformFactory:
    """
    A static factory class to encapsulate data augmentation and normalization strategies.
    """
    @staticmethod
    def create_standard_transforms(size: int):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Maps to [-1, 1]
        ])

    @staticmethod
    def create_mask_transforms(size: int):
        return transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])