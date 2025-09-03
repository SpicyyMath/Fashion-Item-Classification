import torch
import torch.nn as nn
import torchvision.models as models

def pretrained_resnet_model(num_classes, dropout_p=0.3):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    if dropout_p > 0:
        print(f"Adding Dropout layer with p={dropout_p} before the final Linear layer.")
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        print("No dropout layer added (dropout_p <= 0).")
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def main():
    num_classes = 10
    pretrained_model = pretrained_resnet_model(num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)

    print(pretrained_model)

if __name__ == '__main__':
    main()