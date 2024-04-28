import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchviz import make_dot
from graphviz import Digraph

import torch
import torch.nn as nn

class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)

    def forward_state(self, state):
        return self.state_embedding(state)

    def forward_action(self, action):
        return self.action_embedding(action)

class MaskedActionPredictionHead(nn.Module):
    def __init__(self, embed_dim, action_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, action_dim)

    def forward(self, x, unmasked_indices, masked_action_indices):
        # Transpose x to match [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)  # Now x is [seq_length*2, batch_size, embed_dim]

        # Select only the unmasked and masked parts of the input sequence
        unmasked_inputs = x[unmasked_indices, :]
        masked_action_inputs = x[masked_action_indices, :]

        # Apply attention
        attn_output, _ = self.attention(masked_action_inputs, unmasked_inputs, unmasked_inputs)

        # Transpose the output back to [batch_size, seq_length, embed_dim] for the linear layer
        attn_output = attn_output.transpose(0, 1)  # Back to [batch_size, seq_length, embed_dim]

        # The outputs from attention are directly related to masked action positions
        predictions = self.output_layer(attn_output)

        return predictions
    

class ForwardDynamicHead(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x, context_indices, target_index):
        # Assume x has already been transposed appropriately before calling
        # x should be [seq_length, batch_size, embed_dim], seq_length should be 3 in this case

        # Apply attention where the last position is the target, and the first two are context
        target = x[target_index, :].unsqueeze(0)  # Add seq_len dimension back
        context = x[context_indices, :]

        # Attention operation
        attn_output, _ = self.attention(target, context, context)

        # Output processing
        predictions = self.output_layer(attn_output.squeeze(0))  # Remove seq_len dimension for linear layer

        return predictions

    
class InverseDynamicHead(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x, context_indices, target_index):
        # Assume x has already been transposed appropriately before calling
        # x should be [seq_length, batch_size, embed_dim], seq_length should be 3 in this case

        # Apply attention where the last position is the target, and the first two are context
        target = x[target_index, :].unsqueeze(0)  # Add seq_len dimension back
        context = x[context_indices, :]

        # Attention operation
        attn_output, _ = self.attention(target, context, context)

        # Output processing
        predictions = self.output_layer(attn_output.squeeze(0))  # Remove seq_len dimension for linear layer

        return predictions


class TransformerPredictor(nn.Module):
    def __init__(self, state_embed_dim, state_dim, action_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = StateActionEmbedding(state_dim=state_dim, action_dim=action_dim, embed_dim=state_embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(state_embed_dim, action_dim)
        #self.dynamic_prediction = nn.Linear(state_embed_dim * 2, state_dim)
        #self.inverse_prediction = nn.Linear(state_embed_dim * 2, action_dim)
        self.forward_dynamic_head = ForwardDynamicHead(state_embed_dim, state_dim, num_heads)
        self.inverse_dynamic_head = InverseDynamicHead(state_embed_dim, action_dim, num_heads)
        self.masked_action_head = MaskedActionPredictionHead(state_embed_dim, action_dim, num_heads)

    def forward(self, states, actions):
        mask_indices_states, mask_indices_actions = self.random_mask(states, actions)
        seq_length = states.shape[1]
        embed = torch.zeros(states.shape[0], seq_length*2, state_embed_dim, device=states.device)

        for i in range(seq_length):
            embed[:, i*2, :] = self.embedding.forward_state(states[:, i, :])
            embed[:, i*2+1, :] = self.embedding.forward_action(actions[:, i, :])

        transformer_output = self.transformer(embed)
        
        action_prediction = self.output_layer(transformer_output[:, -1, :])  # Example for last action
        #forward_prediction = self.dynamic_prediction(torch.cat((transformer_output[:, -2, :], transformer_output[:, -1, :]), dim=-1))
        #inverse_prediction = self.inverse_prediction(torch.cat((transformer_output[:, -2, :], transformer_output[:, 0, :]), dim=-1))

        Dynamic_forward_predictions = torch.zeros(states.shape[1]-1, states.shape[0], state_dim, device=states.device)
        Dynamic_inverse_predictions = torch.zeros(states.shape[1]-1, states.shape[0], action_dim, device=states.device)
        counter = 0 
        for i in range(2, seq_length*2, 2):
            embed_slice = embed[:, [i-2, i-1, i], :].transpose(0, 1)  # Transpose to [3, batch_size, embed_dim]
            Dynamic_forward_predictions[counter] = self.forward_dynamic_head(embed_slice, [0, 1], 2)
            counter += 1

        counter = 0
        for i in range(2, seq_length*2, 2):
            embed_slice = embed[:, [i-2, i-1, i], :].transpose(0, 1)  # Transpose to [3, batch_size, embed_dim]
            counter += 1



        # Calculate indices for actions and concatenate with state indices


        combined_indices = torch.cat((mask_indices_states, (mask_indices_actions * 2) + 1), dim=0)

        unmasked = [i for i in range(0, len(embed)) if i not in combined_indices]

        # Use these indices to select the corresponding embeddings
        masked_action_predictions = self.masked_action_head(embed, unmasked, (mask_indices_actions * 2) + 1)

        return action_prediction, Dynamic_forward_predictions, Dynamic_inverse_predictions, masked_action_predictions, mask_indices_states, mask_indices_actions

    def random_mask(self, states, actions):
        seq_length = states.shape[1]
        mask_size_states = seq_length // 2 + 1
        mask_size_actions = seq_length // 2 - 1

        mask_indices_states = torch.randperm(seq_length)[:mask_size_states]
        mask_indices_actions = torch.randperm(seq_length)[:mask_size_actions]

        return mask_indices_states, mask_indices_actions

    def random_mask(self, states, actions):
        seq_length = states.shape[1]
        mask_size_states = seq_length // 2 + 1
        mask_size_actions = seq_length // 2 - 1

        mask_indices_states = torch.randperm(seq_length)[:mask_size_states]
        mask_indices_actions = torch.randperm(seq_length)[:mask_size_actions]

        return mask_indices_states, mask_indices_actions

    def random_mask(self, states, actions):
        seq_length = states.shape[1]
        mask_size_states = seq_length // 2 + 1
        mask_size_actions = seq_length // 2 - 1

        mask_indices_states = torch.randperm(seq_length)[:mask_size_states]
        mask_indices_actions = torch.randperm(seq_length)[:mask_size_actions]

        return mask_indices_states, mask_indices_actions


def train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path=None):
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for i, (state_batch, action_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            action_predictions, forward_predictions, inverse_predictions, masked_action_predictions, mask_indices_states, mask_indices_actions = model(state_batch, action_batch)

            action_targets = action_batch[:, -1, :]
            action_loss = loss_fn(action_predictions, action_targets)

            forward_targets = state_batch[:,:-1:,].transpose(0, 1)     
            forward_loss = loss_fn(forward_predictions, forward_targets)

            inverse_targets = action_batch[:, :-1, :].transpose(0, 1) 
            inverse_loss = loss_fn(inverse_predictions, inverse_targets)
            
            # Gather using created indices
            masked_action_targets = action_batch[:,mask_indices_actions,:]

            masked_action_loss = loss_fn(masked_action_predictions, masked_action_targets)

            total_batch_loss = action_loss + forward_loss + inverse_loss + masked_action_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")

        if save_path:
            torch.save(model.state_dict(), save_path)



state_embed_dim = 20
action_embed_dim = 20
state_dim = 60
action_dim = 10
seq_length =  100  # Arbitrary sequence length
batch_size = 32  # Define the batch size for DataLoader
num_samples = 1000  # Number of samples in the dataset
num_heads = 2
num_layers = 2

# Example initialization and usage
model = TransformerPredictor(state_embed_dim=state_embed_dim, state_dim=state_dim, action_dim=action_dim, num_heads=num_heads, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


# Generate random data for states, actions, and target actions
states = torch.randn(num_samples, seq_length, state_dim)
actions = torch.randn(num_samples, seq_length, action_dim)

# Create TensorDataset
dataset = TensorDataset(states, actions)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Assuming dataset and DataLoader setup here
train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path='model_checkpoint.pth')

# Use torchviz to create a dot graph of the model.
states = torch.randn(num_samples, 25, state_dim)
actions = torch.randn(num_samples, 25, action_dim)
output = model(states, actions)