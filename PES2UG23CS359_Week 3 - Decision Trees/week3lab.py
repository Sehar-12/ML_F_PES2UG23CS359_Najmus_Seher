import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    """
    y = tensor[:, -1]  # target column
    values, counts = torch.unique(y, return_counts=True)
    probabilities = counts.float() / y.shape[0]

    # Avoid log(0): mask zero probabilities
    nonzero_probs = probabilities[probabilities > 0]
    entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs))
    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    """
    total_rows = tensor.shape[0]
    col = tensor[:, attribute]
    values, counts = torch.unique(col, return_counts=True)

    avg_info = 0.0
    for v, cnt in zip(values, counts):
        subset = tensor[col == v]
        weight = cnt.item() / total_rows
        entropy_sv = get_entropy_of_dataset(subset)
        avg_info += weight * entropy_sv
    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute (rounded to 4 decimals).
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    ig = dataset_entropy - avg_info
    return round(float(ig), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    """
    num_attributes = tensor.shape[1] - 1  # exclude target
    info_gains = {}

    for attr in range(num_attributes):
        ig = get_information_gain(tensor, attr)
        info_gains[attr] = ig

    # Find attribute with maximum information gain
    best_attr = max(info_gains, key=info_gains.get)
    return (info_gains, best_attr)
