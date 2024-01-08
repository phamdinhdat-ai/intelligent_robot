import torch
import torch.nn as nn
import torch.nn.functional as F
class RobotModel(nn.Module):
    
    def __init__(self, input_size=3, hidden_size=25, out_class = 6, dropout = 0.2):
        super(RobotModel, self).__init__()
        
        self.input_size = input_size 
        self.hidden_size  = hidden_size 
        self.out_class  = out_class
        self.dropout =  dropout
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first = True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.out_class)
        
    def forward(self, x):
        
        x, _ = self.rnn(x)
        x = self.sigmoid(x)
        x = self.fc(x)
        out = F.sigmoid(x)
        return out
    
    
# model = RobotModel(input_size=6, hidden_size=25, out_class= 6, dropout=0.2)
# print(model.eval())


# x = torch.randn(100,1, 6)
# out = model(x)
# print(out)
# print(out.shape)