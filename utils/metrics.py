"""Evaluation and metrics utilities for meta-learning experiments."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.vq import kmeans, vq
from torch.func import functional_call

from .loss_utils import compute_geometry_consistency_loss


def get_cluster_assignments(logits: torch.Tensor, targets, num_classes: int) -> torch.Tensor:
    """Compute optimal cluster-to-class assignment using Hungarian matching.
    
    Checks which logits belong to which targets by looking at which neuron
    outputted the most often for a certain class.
    
    Args:
        logits: Model output logits of shape (batch_size, num_classes)
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Permutation tensor for aligning logits with targets
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    probs = F.softmax(logits, dim=1)
    probs = probs.detach().to("cpu").numpy()
    preds = np.argmax(probs, axis=1)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        confusion[p, t] += 1

    class_best = []
    for c in range(num_classes):
        best_neuron = np.argmax(confusion[:, c])
        best_count = confusion[best_neuron, c]
        class_best.append((c, best_neuron, best_count))

    class_best.sort(key=lambda x: x[2], reverse=True)

    perm_indices = [None] * num_classes
    used_neurons = set()

    for c, neuron, _ in class_best:
        if neuron not in used_neurons:
            perm_indices[c] = neuron
            used_neurons.add(neuron)

    remaining_neurons = [n for n in range(num_classes) if n not in used_neurons]
    for c in range(num_classes):
        if perm_indices[c] is None:
            perm_indices[c] = remaining_neurons.pop()

    return torch.tensor(perm_indices, device=logits.device)


def evaluate_classification(
    hyper,
    target,
    dataset_name: str,
    resources,
    distributions_targets: list,
    mu_max_dist,
    config: dict,
    device: torch.device,
):
    """Evaluate classification accuracy with and without inner loop adaptation.
    
    Args:
        hyper: HyperNetwork model
        target: TargetNet model
        dataset_name: Name of dataset to evaluate on
        resources: ResourceManager instance
        distributions_targets: List of (mu, logvar, y) tuples for evaluation
        mu_max_dist: Maximum distance for geometry loss (unused currently)
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        Tuple of (accuracy_no_training, accuracy_with_training)
    """
    num_classes = config["data"]["num_classes"]
    steps_innerloop = config["meta"]["steps_innerloop"]
    lr_inner = float(config["training"]["lr_inner"])
    weight_inner_loss = config["training"]["weight_innerloss"]
    
    vae_distribution = resources.get_vae_data_distribution(dataset_name)
    dist_inner_pool = vae_distribution
    dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]

    params_base = hyper(dist_inner)

    correct_base = 0
    total_base = 0

    with torch.no_grad():
        for mu, logvar, y in distributions_targets:
            y = y.to(device)
            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_base)

            perm = get_cluster_assignments(logits, y, num_classes)
            aligned_logits = logits[:, perm]

            preds = torch.argmax(aligned_logits, dim=1)
            correct_base += (preds == y).sum().item()
            total_base += y.size(0)

    acc_no_training = correct_base / total_base if total_base > 0 else 0.0

    buffers_hyper = dict(hyper.named_buffers())

    correct_adapted = 0
    total_adapted = 0

    for mu, logvar, y in distributions_targets:

        current_params_hyper = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in hyper.named_parameters()
        }

        for _ in range(steps_innerloop):

            dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]

            params_target = functional_call(
                hyper, (current_params_hyper, buffers_hyper), (dist_inner,)
            )

            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_target)
            target_output = F.normalize(logits, p=2, dim=1)

            loss = compute_geometry_consistency_loss(mu, target_output)
            loss = loss * weight_inner_loss

            grads = torch.autograd.grad(
                loss, current_params_hyper.values(), create_graph=False
            )

            current_params_hyper = {
                name: (p - g * lr_inner).detach().requires_grad_(True)
                for (name, p), g in zip(current_params_hyper.items(), grads)
            }

        with torch.no_grad():
            y = y.to(device)
            dist_inner = dist_inner_pool[torch.randperm(dist_inner_pool.size(0))[:1024]]
            params_target_adapted = functional_call(
                hyper, (current_params_hyper, buffers_hyper), (dist_inner,)
            )

            distribution = torch.concat((mu, logvar), dim=1)
            logits = target.forward(distribution, params_target_adapted)

            perm = get_cluster_assignments(logits, y, num_classes)
            aligned_logits = logits[:, perm]

            preds = torch.argmax(aligned_logits, dim=1)
            correct_adapted += (preds == y).sum().item()
            total_adapted += y.size(0)

    acc_training = correct_adapted / total_adapted if total_adapted > 0 else 0.0

    return acc_no_training, acc_training


def get_kmeans_accuracy(features: np.ndarray, targets: np.ndarray, k: int, iterations: int) -> float:
    """Compute KMeans clustering accuracy.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        targets: Ground truth labels
        k: Number of clusters
        iterations: KMeans iterations
        
    Returns:
        Accuracy after optimal cluster-to-class assignment
    """
    centroids, _ = kmeans(features, k, iter=iterations)
    cluster_ids, _ = vq(features, centroids)  

    cluster_ids_t = torch.from_numpy(cluster_ids).long().to("cpu")
    logits = F.one_hot(cluster_ids_t, num_classes=k).float()

    perm = get_cluster_assignments(logits=logits, targets=targets, num_classes=k)

    aligned_logits = logits[:, perm]

    preds = aligned_logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()

    return acc
