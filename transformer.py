import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

state_embed_dim = 20
action_embed_dim = 20

class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim):
        super(StateActionEmbedding, self).__init__()
        # Define separate linear layers for state and action
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
    
    def forward_state(self, state):
        # Flatten and embed the state
        state = state.view(state.size(0), -1)  # Reshape to (batch_size, 600)
        return self.state_embed(state)

    def forward_action(self, action):
        # Flatten and embed the action
        action = action.view(action.size(0), -1)  # Reshape to (batch_size, 100)
        return self.action_embed(action)

class TransformerPredictor(nn.Module):
    def __init__(self, state_embed_dim, action_embed_dim, num_heads, num_layers, num_actions):
        super(TransformerPredictor, self).__init__()
        self.embedding = StateActionEmbedding(state_dim=10*60, action_dim=10*10, embed_dim=state_embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_embed_dim , nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(state_embed_dim, num_actions)

    def forward(self, states, actions):
        seq_length = states.shape[1]
        embedded = torch.zeros(states.shape[0], seq_length, state_embed_dim + action_embed_dim, device=states.device)
        
        # for i in range(seq_length):
        #     state_embedded = self.embedding.forward_state(states[:, i, :])
        #     action_embedded = self.embedding.forward_action(actions[:, i, :])
        #     embedded[:, i, :] = torch.cat((state_embedded, action_embedded), dim=-1)
        

        embed2= torch.zeros(states.shape[0], seq_length*2, state_embed_dim, device=states.device)

        for i in range(seq_length):
            print( i*2, (i*2)+1)
            embed2[:, i*2, :] = self.embedding.forward_state(states[:, i, :])
            embed2[:, (i*2)+1, :] = self.embedding.forward_action(actions[:, i, :])

        transformer_output = self.transformer(embed2[:-1])
        action_predictions = self.output_layer(transformer_output[:, -1, :])
        return action_predictions

def train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path=None):
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for i, (state_batch, action_batch, target_action_batch) in enumerate(dataloader):
            print(f"\nBatch {i+1}")
            print(f"State batch shape: {state_batch.shape}")  # [batch_size, seq_length, state_dim]
            print(f"Action batch shape: {action_batch.shape}")  # [batch_size, seq_length, action_dim]
            print(f"Target action batch shape: {target_action_batch.shape}")  # [batch_size, seq_length, num_actions]

            optimizer.zero_grad()
            action_predictions = model(state_batch, action_batch)
            print(f"Action predictions shape: {action_predictions.shape}")  # [batch_size, num_actions]

            # Select the last timestep's target actions for loss calculation
            targets = target_action_batch[:, -1, :]
            print(f"Targets for loss calculation shape: {targets.shape}")  # [batch_size, num_actions]

            loss = loss_fn(action_predictions, targets[:-1])
            print(f"Loss: {loss.item()}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}, Average Loss: {average_loss}')

        if save_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, save_path)

def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return start_epoch, loss

# Define dimensions and sequence length
state_dim = 600
action_dim = 100
num_actions = 1
seq_length = state_embed_dim-1  # Arbitrary sequence length
batch_size = 32  # Define the batch size for DataLoader
num_samples = 100  # Number of samples in the dataset
num_heads = 2
num_layers = 2

# Example initialization and usage
model = TransformerPredictor(state_embed_dim=state_embed_dim, action_embed_dim=action_embed_dim, num_heads=num_heads, num_layers=num_layers, num_actions=num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


# Generate random data for states, actions, and target actions
states = torch.randn(num_samples, seq_length, state_dim)
actions = torch.randn(num_samples, seq_length, action_dim)
target_actions = torch.randn(num_samples, seq_length, num_actions)

# Create TensorDataset
dataset = TensorDataset(states, actions, target_actions)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Assuming dataset and DataLoader setup here
train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path='model_checkpoint.pth')

# Example loading (you need to create the optimizer before calling this)
# start_epoch, loss = load_model(model, optimizer, 'model_checkpoint.pth')
