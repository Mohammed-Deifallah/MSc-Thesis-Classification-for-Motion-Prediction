from typing import Tuple, Union, Callable

import torch
from torch import nn
from torch.nn import functional as f
import numpy as np

def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    This function is based on by https://github.com/nutonomy/nuscenes-devkit
    Computes the index of the closest trajectory in the lattice as measured by l2 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()



class ConstantLatticeLoss:
    """This class is based on by https://github.com/nutonomy/nuscenes-devkit
    Computes the loss for a constant lattice CoverNet model.
    """
    
    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(np.array(lattice))
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):
            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)
            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()
    

class ADE:
    """This class is inspired by by https://github.com/nutonomy/nuscenes-devkit
    Computes the Average Distance Error for CoverNet model.
    """
    
    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(np.array(lattice))
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):
            
            pred = self.lattice[torch.argmax(logit)]
            closest_lattice_trajectory = self.lattice[self.similarity_func(self.lattice, ground_truth)]
            errors = torch.linalg.norm((closest_lattice_trajectory - pred), ord=2, dim=-1)
            classification_loss = torch.div(torch.sum(errors), len(errors))
            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()

class FDE:
    """This class is inspired by by https://github.com/nutonomy/nuscenes-devkit
    Computes the Final Distance Error for CoverNet model.
    """
    
    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(np.array(lattice))
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Final loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):
            
            pred = self.lattice[torch.argmax(logit)]
            closest_lattice_trajectory = self.lattice[self.similarity_func(self.lattice, ground_truth)]
            classification_loss = torch.linalg.norm((closest_lattice_trajectory[-1] - pred[-1]), ord=2)
            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()