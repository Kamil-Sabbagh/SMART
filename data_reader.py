import os
import json
import random

class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        self.file_index = 0

    def _load_json_file(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def _extract_sequences(self, data):
        #sequences = {}
        observations = []
        actions = []

        for step in data.values():
            current_observation = step["current_observation"]
            action = step["action"]
            
            for agent_index in range(len(current_observation)):
                if len(actions) <= agent_index:
                    actions.append([])
                    observations.append([])
                
                actions[agent_index].append(action[0][agent_index])
                observations[agent_index].append(current_observation[agent_index])
        
        return observations, actions

    def get_next_file(self):
        if self.file_index >= len(self.file_names):
            self.file_index = 0
        
        file_name = self.file_names[self.file_index]
        self.file_index += 1
        
        file_path = os.path.join(self.folder_path, file_name)
        data = self._load_json_file(file_path)
        print
        observations, actions = self._extract_sequences(data)
        
        return observations, actions

# Example usage
# folder_path = 'log_data_episodes'
# data_loader = DataLoader(folder_path)

# # Call the data loader to get the next file's sequences
# observations, actions = data_loader.get_next_file()
# print(len(observations))
# print(len(actions))
