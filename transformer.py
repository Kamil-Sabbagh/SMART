import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchviz import make_dot
from graphviz import Digraph



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
    def __init__(self, state_embed_dim, state_dim, action_dim, num_heads, num_layers, num_actions):
        super(TransformerPredictor, self).__init__()
        self.embedding = StateActionEmbedding(state_dim=state_dim, action_dim=action_dim, embed_dim=state_embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(state_embed_dim, action_dim)  # Corrected to use the proper dimensions
        self.DynamicForwardPredection = nn.Linear((state_embed_dim * 2), state_dim)

    def forward(self, states, actions):
        seq_length = states.shape[1]
        # Correct embedding size to match transformer input
        embed2 = torch.zeros(states.shape[0], seq_length*2, state_embed_dim, device=states.device)  # Assuming embed_dim is available

        for i in range(seq_length):
            embed2[:, i*2, :] = self.embedding.forward_state(states[:, i, :])
            embed2[:, (i*2)+1, :] = self.embedding.forward_action(actions[:, i, :])

        transformer_output = self.transformer(embed2[:,-1,:])
        action_predictions = self.output_layer(transformer_output)

        # Use custom attention to predict next state using the first state and first action
        first_state_embedded = embed2[:, 0, :]  # First state embedding
        first_action_embedded = embed2[:, 1, :]  # First action embedding
        next_state_prediction = self.DynamicForwardPredection(torch.cat((first_state_embedded, first_action_embedded), dim=-1))

        return action_predictions, next_state_prediction



def train_model(model, dataloader, optimizer, action_loss_fn, state_loss_fn, epochs=10, save_path=None):
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for i, (state_batch, action_batch) in enumerate(dataloader):
            #print(f"\nBatch {i+1}")
            #print(f"State batch shape: {state_batch.shape}")  # [batch_size, seq_length, state_dim]
            #print(f"Action batch shape: {action_batch.shape}")  # [batch_size, seq_length, action_dim]

            # Forward pass, now expecting two outputs
            optimizer.zero_grad()
            action_predictions, next_state_predictions = model(state_batch, action_batch)
            

            # For actions, typically you predict the last action
            action_targets = action_batch[:, -1, :]
            #print("Action prediction shape: ", action_predictions.shape)
            #print("Action targer shape: ", action_targets.shape)
            action_loss = action_loss_fn(action_predictions, action_targets)

            # For states, predict the second state in the sequence (as per your design)
            state_targets = state_batch[:, 2, :]  # assuming state_batch shape accommodates this indexing
            #print(next_state_predictions.shape)
            #print(state_targets.shape)
            state_loss = state_loss_fn(next_state_predictions, state_targets)


            # Combine the losses
            combined_loss = action_loss + state_loss
            #print(f"Action Loss: {action_loss.item()}, State Loss: {state_loss.item()}")

            combined_loss.backward()
            optimizer.step()
            total_loss += combined_loss.item()

            #print(f"Epoch {epoch+1} Total Loss: {total_loss}")

            if save_path:
                torch.save(model.state_dict(), save_path)
        
        average_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}, Average Loss: {average_loss}')

    print("Training complete")



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

state_embed_dim = 20
action_embed_dim = 20
state_dim = 60
action_dim = 10
num_actions = 1
seq_length =  100  # Arbitrary sequence length
batch_size = 32  # Define the batch size for DataLoader
num_samples = 1000  # Number of samples in the dataset
num_heads = 2
num_layers = 2

# Example initialization and usage
model = TransformerPredictor(state_embed_dim=state_embed_dim, state_dim=state_dim, action_dim=action_dim, num_heads=num_heads, num_layers=num_layers, num_actions=num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
loss_fn2 = torch.nn.MSELoss()


# Generate random data for states, actions, and target actions
states = torch.randn(num_samples, seq_length, state_dim)
actions = torch.randn(num_samples, seq_length, action_dim)

# Create TensorDataset
dataset = TensorDataset(states, actions)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Assuming dataset and DataLoader setup here
train_model(model, dataloader, optimizer, loss_fn, loss_fn2, epochs=10, save_path='model_checkpoint.pth')

# Use torchviz to create a dot graph of the model.
states = torch.randn(num_samples, 25, state_dim)
actions = torch.randn(num_samples, 25, action_dim)
output = model(states, actions)

# Create a simplified graph
graph = Digraph()
graph.node("Input", "Input (States & Actions)")
graph.node("Embedding", "Embedding Layer")
graph.node("Transformer", "Transformer Encoder")
graph.node("Output", "Output Layer")
graph.node("Pred", "Predictions")

graph.edge("Input", "Embedding")
graph.edge("Embedding", "Transformer")
graph.edge("Transformer", "Output")
graph.edge("Output", "Pred")

# Render the simplified graph
graph.render('simplified_transformer_model', format='png')


# Example loading (you need to create the optimizer before calling this)
# start_epoch, loss = load_model(model, optimizer, 'model_checkpoint.pth')
