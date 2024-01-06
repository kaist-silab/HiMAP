import torch
import torch.nn as nn



def normalized_columns_initializer(std=1.0):
    def _initializer(module):
        if hasattr(module, 'weight'):
            torch.nn.init.orthogonal_(module.weight.data)  # Initialize weight with orthogonal initialization
            module.weight.data *= std

        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)  # Initialize bias to zeros

    return _initializer

class VGG_Net(nn.Module):
    """VGG Network
    
    Args:
        a_size: The number of actions
        rnn_size: The number of hidden units in RNN
        goal_repr_size: The size of goal representation
        keep_prob1: The keep probability of dropout layer 1
        keep_prob2: The keep probability of dropout layer 2
        softmax_temperature: The temperature of softmax layer. Low temperature leads to low exploration (more greedy)
            while high temperature leads to high exploration (more random).
    """
    def __init__(self, 
                 a_size,
                 rnn_size=512,
                 goal_repr_size=12,
                 keep_prob1=1,
                 keep_prob2=1,
                 softmax_temperature=1.0
        ):
        
        super(VGG_Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, rnn_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(rnn_size // 4, rnn_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(rnn_size // 2, rnn_size - goal_repr_size, kernel_size=2),
            nn.ReLU()
        )
        
        self.fc_goal = nn.Linear(3, goal_repr_size)
        
        self.fc1 = nn.Sequential(          
            nn.Linear(rnn_size, rnn_size),
            nn.ReLU(),
            nn.Dropout(keep_prob1)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.ReLU(),
            nn.Dropout(keep_prob2)
        )
        
        self.lstm = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, batch_first=True)

        self.policy_layer = nn.Linear(rnn_size, a_size)
        self.policy_layer.apply(normalized_columns_initializer(std=1./float(a_size)))  # Apply the custom initializer

        self.softmax_temperature = softmax_temperature
        
    def forward(self, observation, goal_pos): 
        conv1 = self.conv1(observation)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  
        
        flat = conv3.view(conv3.size(0), -1)
        goal_layer = self.fc_goal(goal_pos)
        
        hidden_input = torch.cat((flat, goal_layer), dim=1)
        
        h1 = self.fc1(hidden_input)
        h2 = self.fc2(h1)
        self.h3 = h2 + hidden_input
        self.h3 = self.h3.unsqueeze(1)
        output,_ = self.lstm(self.h3)
        output = output.squeeze(1)  # can't omit the number in the bracket, otherwise error will occur in softmax process when batch size is 1
        
        policy_layer = self.policy_layer(output)
                
        # Use softmax temperature to control the exploration
        policy = torch.softmax(policy_layer / self.softmax_temperature, dim=1)
        
        return policy