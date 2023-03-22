import torch

class BlindClassifier(torch.nn.Module):
    """ 
    A blind classifier that returns a random label
    base on previously seen labels. It samples from
    a multinomial distribution with the probability
    of each label being the fraction of times that
    label has been seen so far.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.label_counts = torch.zeros(num_classes)

    def forward(self, x):
        # Returns a sample of batch size x.shape[0] from the multinomial distribution
        # with the probability of each label being the fraction of times that label
        # has been seen so far.
        return torch.multinomial(self.label_counts, x.shape[0], replacement=False).to(x.device)

    def update(self, labels):
        # Update the label counts
        for label in labels:
            self.label_counts[label] += 1

    def reset(self):
        self.label_counts = torch.zeros(self.num_classes)

    def __repr__(self):
        return f"BlindClassifier(num_classes={self.num_classes})"
