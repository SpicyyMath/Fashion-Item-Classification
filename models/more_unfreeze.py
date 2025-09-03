import torch
import torch.nn as nn
import torchvision.models as models

def pretrained_resnet_model_flexible_freeze(num_classes, freeze_until=None, dropout_p=0.3):
    """
    Loads a pretrained ResNet-18 model, allows flexible freezing of layers,
    and replaces the final fully connected layer with a new one (optionally
    preceded by dropout) for the specified number of classes.

    Args:
        num_classes (int): The number of output classes for the new classifier.
        freeze_until (str, optional): The name of the layer *before* which
            all layers should be frozen. If None, no layers are frozen
            (except potentially by default pretraining settings). Example values:
            'layer1', 'layer2', 'layer3', 'layer4'. If the name is not found,
            all backbone layers might be frozen. Defaults to None.
        dropout_p (float, optional): The probability for the dropout layer
            added before the final fully connected layer. If 0 or less,
            no dropout layer is added. Defaults to 0.5.

    Returns:
        torch.nn.Module: The modified ResNet model.
    """
    model = models.resnet18(pretrained=True)
    print(f"Loaded pretrained ResNet-18.")

    if freeze_until:
        print(f"Freezing layers up to (but not including): {freeze_until}")
        freeze_layers = True
        for name, param in model.named_parameters():
            current_layer_block_name = name.split('.')[0]

            if freeze_layers and current_layer_block_name == freeze_until:
                print(f"--- Reached {freeze_until}, stopping freeze ---")
                freeze_layers = False
            if freeze_layers and not name.startswith('fc.'):
                param.requires_grad = False
            else:
                param.requires_grad = True


        if not freeze_layers:
            print(f"Layers from {freeze_until} onwards (and fc) will be trained.")
        else:
            print(f"All backbone layers frozen (or '{freeze_until}' not found/matched well in backbone). Only fc will be trained initially.")

            for name, param in model.named_parameters():
                if not name.startswith('fc.'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    else:
        print("No layer freezing requested (freeze_until=None). All layers trainable initially.")
        for param in model.parameters():
            param.requires_grad = True

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

    print(f"Replaced original fc layer with a new classifier for {num_classes} classes.")
    # Check requires_grad status of the *new* fc layer's parameters
    # Note: If using Sequential, the linear layer is now model.fc[1]
    final_classifier_layer = model.fc[-1] if isinstance(model.fc, nn.Sequential) else model.fc
    print(f"New final classifier parameters require_grad: {final_classifier_layer.weight.requires_grad}")

    print("\nParameter requires_grad status summary:")
    for name, param in model.named_parameters():
        is_trainable = param.requires_grad
        # Add indicator if it's part of the new fc module
        fc_marker = " (in new fc)" if name.startswith("fc.") else ""
        status = "Trainable" if is_trainable else "Frozen"
        print(f"  {status}: {name}{fc_marker}")

    return model

