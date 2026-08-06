import torchvision.transforms as transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training data augmentation pipeline for medical images.
    
    Applies random resized crop, horizontal flip, color jitter, and normalization.
    
    Args:
        image_size: Target image spatial size.
        
    Returns:
        Composed transform pipeline for training.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transform pipeline (no augmentation).
    
    Args:
        image_size: Target image spatial size.
        
    Returns:
        Composed transform pipeline for validation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
