import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from data_reader import DataLoader as CustomDataLoader
from model import TransformerPredictor

def plot_metrics(metrics, steps, save_path='plots'):
    """
    Plots training metrics and saves the plots as images.

    Args:
        metrics (dict): A dictionary containing lists of metric values.
        steps (list): A list of step indices corresponding to the metric values.
        save_path (str): The directory path where the plots will be saved.
    """
    import matplotlib.pyplot as plt  # Import here to avoid unnecessary import if the function is not used

    plt.figure(figsize=(12, 8))

    # Plot Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(steps, metrics['accuracy'], label='Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Forward Loss
    plt.subplot(2, 2, 2)
    plt.plot(steps, metrics['forward_loss'], label='Forward Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Inverse Loss
    plt.subplot(2, 2, 3)
    plt.plot(steps, metrics['inverse_loss'], label='Inverse Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Ensure plots are neatly organized and save to the specified path
    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}/metrics_step_{steps[-1]}.png')
    plt.close()

def train_model(model, custom_dataloader, optimizer, action_loss_fn, state_loss_fn, epochs=10, og_window_size=0, save_path=None):
    """
    Trains the model using data from the custom dataloader.

    Args:
        model (nn.Module): The model to be trained.
        custom_dataloader (CustomDataLoader): The custom data loader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        action_loss_fn (nn.Module): Loss function for action predictions.
        state_loss_fn (nn.Module): Loss function for state predictions.
        epochs (int, optional): Number of epochs to train for. Default is 10.
        og_window_size (int, optional): Original window size for the moving window approach. Default is 0.
        save_path (str, optional): Path to save the model. Default is None.

    Returns:
        tuple: Contains two lists, metrics and steps, representing training metrics and corresponding steps.
    """
    writer = SummaryWriter()  # Initialize TensorBoard writer
    metrics = {'accuracy': [], 'forward_loss': [], 'inverse_loss': []}
    steps = []

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Step learning rate scheduler to reduce learning rate after 10 epochs
    model.train()  # Set model to training mode

    # To store metrics for the last 500 steps for more stable average calculation
    last_500_accuracies = []
    last_500_forward_losses = []
    last_500_inverse_losses = []

    for epoch in range(epochs):
        total_batches = 0  # Track the total number of batches processed

        # Iterate through all files provided by the custom data loader
        for file_index in tqdm(range(len(custom_dataloader.file_names)), desc=f"Epoch {epoch+1}/{epochs}"):
            states, actions = custom_dataloader.get_next_file()

            # Iterate through all agents in the current file
            for agent_index in states:
                state_data = torch.tensor(states[agent_index], dtype=torch.float32).unsqueeze(0)
                action_data = torch.tensor(actions[agent_index], dtype=torch.long).unsqueeze(0).unsqueeze(-1)

                # Determine the window size for the moving window approach
                window_size = og_window_size if og_window_size == 0 or og_window_size > action_data.shape[1] else action_data.shape[1]

                # Apply the moving window approach to each agent's data
                for m in range(0, action_data.shape[1]):
                    if m + window_size + 1 > state_data.shape[1] or m + window_size > action_data.shape[1]:
                        break

                    # Extract batches for the current window
                    state_batch = state_data[:, m:m + window_size + 1, :]
                    action_batch = action_data[:, m:m + window_size, :]

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    outputs = model(state_batch, action_batch)

                    # Unpack model outputs
                    action_predictions, forward_predictions, inverse_predictions, *_ = outputs

                    # Calculate losses
                    action_targets = action_batch[:, -1].view(-1).long()
                    action_loss = action_loss_fn(action_predictions, action_targets)

                    forward_targets = state_batch[:, 1:, :].transpose(0, 1).float()
                    forward_loss = state_loss_fn(forward_predictions, forward_targets)

                    inverse_targets = action_batch[:, :-1].view(-1)
                    inverse_loss = action_loss_fn(inverse_predictions.view(-1, inverse_predictions.size(-1)), inverse_targets)

                    # Total loss
                    total_batch_loss = action_loss + forward_loss + inverse_loss
                    total_batch_loss.backward()  # Backward pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()  # Update model parameters

                    # Calculate action prediction accuracy
                    action_accuracy = (action_predictions.argmax(dim=-1).view(-1) == action_targets).float().mean().item() * 100

                    # Append metrics for the last 500 steps
                    last_500_accuracies.append(action_accuracy)
                    last_500_forward_losses.append(forward_loss.item())
                    last_500_inverse_losses.append(inverse_loss.item())

                    # Maintain only the last 500 entries for each metric
                    if len(last_500_accuracies) > 500:
                        last_500_accuracies.pop(0)
                        last_500_forward_losses.pop(0)
                        last_500_inverse_losses.pop(0)

                    total_batches += 1

                    # Print and log metrics every 500 steps
                    if total_batches % 500 == 0:
                        avg_accuracy = sum(last_500_accuracies) / len(last_500_accuracies)
                        avg_forward_loss = sum(last_500_forward_losses) / len(last_500_forward_losses)
                        avg_inverse_loss = sum(last_500_inverse_losses) / len(last_500_inverse_losses)

                        metrics['accuracy'].append(avg_accuracy)
                        metrics['forward_loss'].append(avg_forward_loss)
                        metrics['inverse_loss'].append(avg_inverse_loss)
                        steps.append(total_batches)

                        # Log metrics to TensorBoard
                        writer.add_scalar('Accuracy/train', avg_accuracy, total_batches)
                        writer.add_scalar('Forward Loss/train', avg_forward_loss, total_batches)
                        writer.add_scalar('Inverse Loss/train', avg_inverse_loss, total_batches)
                        
                        tqdm.write(f"Step {total_batches}, Accuracy: {avg_accuracy:.4f}%, Total Loss: {total_batch_loss:.4f}, Forward Loss: {avg_forward_loss:.4f}, Inverse Loss: {avg_inverse_loss:.4f}")

                        # Save model checkpoint
                        if save_path:
                            torch.save(model.state_dict(), f"{save_path}/smart_transformer.pth")
                            print(f"The model has been saved to path: {save_path}/smart_transformer.pth")

        scheduler.step()  # Step the learning rate scheduler

    writer.close()  # Close the TensorBoard writer
    return metrics, steps


# Define folder path and data loader
folder_path = 'log_data_episodes'
custom_dataloader = CustomDataLoader(folder_path)

# Model hyperparameters
state_embed_dim = 20
state_dim = 2  # Example number of state dimensions
action_dim = 1  # Example number of action dimensions
num_heads = 2
num_layers = 2
num_action_classes = 5

# Initialize model, optimizer, and loss functions
model = TransformerPredictor(
    state_embed_dim=state_embed_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    num_action_classes=num_action_classes,
    num_heads=num_heads,
    num_layers=num_layers
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
action_loss_fn = nn.CrossEntropyLoss()
state_loss_fn = nn.MSELoss()

# Train the model and capture metrics
metrics, steps = train_model(
    model,
    custom_dataloader,
    optimizer,
    action_loss_fn,
    state_loss_fn,
    epochs=10,
    og_window_size=0,
    save_path='saved_models/'
)

print(metrics)
