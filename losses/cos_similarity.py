import torch
from torch import nn
class Cosine_loss(nn.Module):
    def __init__(self,margin=-0.5):
        super(Cosine_loss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.margin = margin  
    
    def forward(self, x, y):
        # x : [B D]
        # y : [B D]
        #print(self.margin)
        similarity_matrix = self.cosine(x, y)
        #print(similarity_matrix)
        #similarity_loss = torch.mean(torch.relu(self.margin - similarity_matrix))
        similarity_loss = torch.mean(torch.relu(similarity_matrix - self.margin ))
        return similarity_loss