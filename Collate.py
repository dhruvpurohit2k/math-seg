import torch


class Collate:
    def __init__(self, pad_idx, target_height=96):
        self.pad_idx = pad_idx
        self.target_height = target_height

    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Pad images
        max_width = max([img.shape[2] for img in images])
        padded_images = torch.ones(len(images), 1, self.target_height, max_width)
        for i, img in enumerate(images):
            padded_images[i, :, :, : img.shape[2]] = img

        # Pad labels
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.pad_idx
        )

        lengths = [len(lab) for lab in labels]

        return padded_images, padded_labels, lengths
