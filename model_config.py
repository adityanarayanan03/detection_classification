from torchvision import models
import torch.nn as nn


ml_og_req_grads = []
def make_ml_model():
    global ml_og_req_grads
    model = models.resnet18(pretrained=True)
    y_dim = 3
    model.fc = nn.Sequential(nn.Dropout(0.4),
                             nn.Linear(model.fc.in_features, y_dim),
                             )
    ml_og_req_grads = [p.requires_grad for p in model.parameters()]
    
    return model