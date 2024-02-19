from torch import nn
import torchvision 


def create_model():
    weight = torchvision.models.resnet50.weight = 'DEFAULT'
    model = torchvision.models.resnet50(weights = weight)
    for param in model.parameters():
        param.requires_grad=False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , 256)
    model.fc = nn.Flatten()
    model.fc = nn.Linear(num_features , 128)
    model.fc = nn.Flatten()
    model.fc = nn.Linear(num_features , 64)
    model.fc = nn.Flatten()
    model.fc = nn.Linear(num_features , 8)
    model.fc = nn.Flatten()
    model.fc = nn.Linear(num_features , 2)
    return model
