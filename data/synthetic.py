import torch


def create_synthetic_dataset(n_samples: int, config: dict):
    """Create a synthetic dataset for smoke testing when real data is unavailable.

    Mirrors the MIMICDataset batch contract:
        image (3, H, W), ehr (num_ehr,), label scalar, ehr_mask scalar,
        img_mask scalar, sample_idx scalar.

    Args:
        n_samples: Number of samples.
        config: Configuration dictionary.

    Returns:
        A torch.utils.data.Dataset over random tensors.
    """
    num_classes = config.get("data", {}).get("num_classes", 5)
    num_ehr = config.get("data", {}).get("num_ehr_features", 12)
    image_size = config.get("data", {}).get("image_size", 224)

    images = torch.randn(n_samples, 3, image_size, image_size)
    ehr = torch.randn(n_samples, num_ehr)
    labels = torch.randint(0, num_classes, (n_samples,))
    ehr_mask = torch.ones(n_samples)
    img_mask = torch.ones(n_samples)

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, images, ehr, labels, ehr_mask, img_mask):
            self.images = images
            self.ehr = ehr
            self.labels = labels
            self.ehr_mask = ehr_mask
            self.img_mask = img_mask

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "image": self.images[idx],
                "ehr": self.ehr[idx],
                "label": self.labels[idx],
                "ehr_mask": self.ehr_mask[idx],
                "img_mask": self.img_mask[idx],
                "sample_idx": idx,
            }

    return SyntheticDataset(images, ehr, labels, ehr_mask, img_mask)
