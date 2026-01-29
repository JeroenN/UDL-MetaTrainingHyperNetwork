"""Meta-training loop for hypernetwork experiments."""
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.func import functional_call
from tqdm import tqdm

from .loss_utils import compute_geometry_consistency_loss
from .metrics import evaluate_classification, get_cluster_assignments, get_kmeans_accuracy
from .vae_utils import get_gaussian_from_vae
from . import plotting


def meta_training(
    hyper,
    target,
    train_dataset_names: list,
    test_dataset_name: str,
    resources,
    config: dict,
    device: torch.device,
    print_grads: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
):
    """Meta-training loop for the hypernetwork.
    
    Args:
        hyper: HyperNetwork model
        target: TargetNet model
        train_dataset_names: List of training dataset names
        test_dataset_name: Name of test dataset for evaluation
        resources: ResourceManager instance
        config: Configuration dictionary with all hyperparameters
        device: Torch device
        print_grads: Whether to print gradient statistics
        output_dir: Directory to save plots and embeddings
    """
    output_dir = Path(output_dir) if output_dir else None
    
    # Extract config values
    lr_outer = float(config["training"]["lr_outer"])
    lr_inner = float(config["training"]["lr_inner"])
    epochs_hyper = config["training"]["epochs_hyper"]
    steps_innerloop = config["meta"]["steps_innerloop"]
    num_classes = config["data"]["num_classes"]
    weight_inner_loss = config["training"]["weight_innerloss"]
    log_interval = config["training"]["log_interval"]
    use_combined_loader = config["data"].get("use_combined_loader", False)
    save_embeddings_interval = config.get("experiment", {}).get("save_embeddings_interval", 10)
    
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr_outer)

    distributions_targets = []
    mus = []
    ys = []
    loader = resources.get_loader(test_dataset_name)
    vae = resources.get_vae(test_dataset_name)
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            mu, logvar = get_gaussian_from_vae(vae, X, 0, visualize=False)
            distributions_targets.append((mu, logvar, y)) 
            ys.append(y)
            mus.append(mu)

            if batch_idx == steps_innerloop:
                break

    kmeans_mus = torch.concat(mus, dim=0).cpu().numpy()
    kmeans_y = torch.concat(ys, dim=0).cpu().numpy()
    kmeans_acc = get_kmeans_accuracy(kmeans_mus, kmeans_y, num_classes, 20)   
    print(f"\n KMEANS ACCURACY: {kmeans_acc} \n")

    inner_losses = defaultdict(list)
    outer_losses = defaultdict(list)
    acc_training_list = []
    acc_no_training_list = []
    average_acc_diff = []
    mu_max_dist = None

    for epoch in tqdm(range(epochs_hyper), desc="Epochs"):
        params_hyper = dict(hyper.named_parameters())
        buffers_hyper = dict(hyper.named_buffers())

        if use_combined_loader:
            vae_dist_inner = resources.get_combined_vae_distribution()
            dataset_name = "combined"
        else:
            # Select a random training subset
            train_subset_names = list(resources.train_datasets.keys())
            dataset_name = random.choice(train_subset_names)
            vae_dist_inner = resources.get_vae_data_distribution(dataset_name)

        adapted_params_hyper = {k: v.clone() for k, v in params_hyper.items()}
        
        # Innerloop
        for _ in range(steps_innerloop):
            if use_combined_loader:
                inner_X, _, inner_mu, inner_logvar = resources.get_combined_batch()
            else:
                inner_X, _, inner_mu, inner_logvar = resources.get_batch(dataset_name)
            dist_inner = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]
            inner_distribution_input = torch.concat((inner_mu, inner_logvar), dim=1)

            params_target_inner = functional_call(
                hyper, (adapted_params_hyper, buffers_hyper), (dist_inner,)
            )

            target_logits = target.forward(
                inner_distribution_input, params_target_inner
            )

            target_output = F.normalize(target_logits, p=2, dim=1)

            inner_loss = compute_geometry_consistency_loss(
                inner_mu, target_output, mu_max_dist
            )
            inner_loss = inner_loss * weight_inner_loss
            grads = torch.autograd.grad(
                inner_loss,
                adapted_params_hyper.values(),
                create_graph=True,
                allow_unused=False,
            )

            adapted_params_hyper = {
                name: p - g * lr_inner
                for (name, p), g in zip(adapted_params_hyper.items(), grads)
            }

        # Outerloop
        dist_outer = vae_dist_inner[torch.randperm(vae_dist_inner.size(0))[:1024]]
        if use_combined_loader:
            X_outer, y_outer, mu_outer, logvar_outer = resources.get_combined_batch()
        else:
            X_outer, y_outer, mu_outer, logvar_outer = resources.get_batch(dataset_name)
        outer_distribution_input = torch.concat((mu_outer, logvar_outer), dim=1)

        params_target_outer = functional_call(
            hyper, (adapted_params_hyper, buffers_hyper), (dist_outer,)
        )

        logits_outer = target.forward(outer_distribution_input, params_target_outer)

        perm = get_cluster_assignments(logits_outer, y_outer, num_classes)

        aligned_logits_outer = logits_outer[:, perm]

        outer_loss = F.cross_entropy(aligned_logits_outer, y_outer)

        optimizer.zero_grad()
        outer_loss.backward()

        if print_grads:
            for name, p in hyper.named_parameters():
                if p.grad is not None:
                    print(
                        f"{name}: grad mean={p.grad.mean():.4e}, max={p.grad.abs().max():.4e}"
                    )

        optimizer.step()

        acc_no_training, acc_training = evaluate_classification(
            hyper,
            target,
            test_dataset_name,
            resources,
            distributions_targets,
            mu_max_dist,
            config,
            device,
        )
        inner_losses[dataset_name].append(inner_loss.detach().item())
        outer_losses[dataset_name].append(outer_loss.detach().item())
        acc_no_training_list.append(acc_no_training)
        acc_training_list.append(acc_training)
        if epoch > 50:
            acc_diff = [
                a_train - a_no_train
                for a_train, a_no_train in zip(acc_training_list, acc_no_training_list)
            ]
            average_acc_diff.append(sum(acc_diff[-50:]) / 50)
        if epoch % log_interval == 0:
            print(f"accuracy no training {acc_no_training:.4f}")
            print(f"accuracy training {acc_training: .4f}")
            if epoch > 100:
                print(
                    f"average acc training last 100 {sum(acc_training_list[-100:])/100:.4f}"
                )
                print(
                    f"average acc no training last 100 {sum(acc_no_training_list[-100:])/100:.4f}"
                )
            print(f"outer_loss: {outer_loss: .4f}")
            print(f"inner_loss: {inner_loss: .4f}")
            if output_dir:
                plotting.plot_losses_and_accuracies(
                    inner_losses, outer_losses, acc_training_list, 
                    average_acc_diff, kmeans_acc, output_dir
                )
        
        # Save embeddings periodically
        if output_dir and save_embeddings_interval > 0 and (epoch + 1) % save_embeddings_interval == 0:
            embeddings_dir = output_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect embeddings from the test dataset
            all_mu = []
            all_logvar = []
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                vae_dist = resources.get_vae_data_distribution(test_dataset_name)
                dist_sample = vae_dist[torch.randperm(vae_dist.size(0))[:1024]]
                params = hyper(dist_sample)
                
                for mu, logvar, y in distributions_targets:
                    distribution = torch.concat((mu, logvar), dim=1)
                    logits = target.forward(distribution, params)
                    
                    all_mu.append(mu.cpu())
                    all_logvar.append(logvar.cpu())
                    all_logits.append(logits.cpu())
                    all_labels.append(y.cpu())
            
            embeddings_data = {
                "epoch": epoch + 1,
                "mu": torch.cat(all_mu, dim=0),
                "logvar": torch.cat(all_logvar, dim=0),
                "logits": torch.cat(all_logits, dim=0),
                "labels": torch.cat(all_labels, dim=0),
            }
            torch.save(embeddings_data, embeddings_dir / f"epoch_{epoch + 1}.pt")
            print(f"Saved embeddings at epoch {epoch + 1}")
