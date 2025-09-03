# --- START OF FILE train.py ---

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# Import Learning Rate Schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import traceback # For printing detailed error messages

# --- Model Imports ---
# Ensure these paths and class names match your project structure
try:
    from models.Newmodel import Newmodel
    from models.more_unfreeze import pretrained_resnet_model_flexible_freeze
    from models.pretrained_model import pretrained_resnet_model
    from models.attention_is_all_you_need import DynamicRELANClassifier
    from models.cluster_model import FeatureExtractor, ClusterModel, Classifier
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please check the 'models' directory and filenames.")
    exit()

# --- Data Loader Import ---
try:
    from utils.data_loader import create_data_loaders, create_data_loaders2, class_to_idx, num_classes, superclass_to_idx, num_classes_superclass
except ImportError:
    print("Error: Could not import from utils/data_loader.py.")
    print("Please ensure data_loader.py exists in the 'utils' subdirectory and defines the required functions/variables.")
    exit()
except NameError as e:
     print(f"Error: A name expected from data_loader.py was not found: {e}")
     exit()


# --- Utility Imports ---
try:
    from utils.train_loss import get_loss_criterion
    from utils.visualize import visualize_loss_curves
except ImportError:
    print("Error: Could not import from utils/train_loss.py or utils/visualize.py.")
    print("Please ensure these files exist in a 'utils' subdirectory.")
    exit()

# ==============================================================================
# train_model Function (Includes Accumulation, Warmup, Scheduler, Clipping)
# ==============================================================================
def train_model(model, train_loader, val_loader, loss_criterion, optimizer, scheduler, num_epochs, device, runs_dir, patience=5, accumulation_steps=1, warmup_epochs=0, clip_grad_value=None):
    """
    Training model function with early stopping, gradient accumulation, LR scheduling, warmup,
    gradient clipping, tqdm progress display, GPU training, and loss recording.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler instance.
                                                                     Needs step() called appropriately.
        num_epochs (int): Maximum number of epochs to train for.
        device (torch.device): The device to train on (e.g., 'cuda:0' or 'cpu').
        runs_dir (str): Directory to save the best model weights.
        patience (int): Early stopping patience (epochs).
        accumulation_steps (int): Gradient accumulation steps.
        warmup_epochs (int): Number of epochs for linear warmup (0 to disable).
        clip_grad_value (float or None): Value for gradient clipping by value. None to disable.

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    best_val_loss = np.inf
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop = False
    Path(runs_dir).mkdir(parents=True, exist_ok=True) # Ensure runs directory exists
    best_model_path = os.path.join(runs_dir, 'best_model.pth')

    start_time = time.time()
    train_losses = []
    val_losses = []
    val_accuracies = []

    initial_lr = optimizer.param_groups[0]['lr'] # Store initial LR for warmup calc

    # --- Print Initial Configuration ---
    print(f"Early stopping patience set to: {patience} epochs")
    if warmup_epochs > 0:
        print(f"Warmup enabled for the first {warmup_epochs} epochs, up to LR {initial_lr:.6f}.")
    if scheduler:
        scheduler_name = scheduler.__class__.__name__
        print(f"Using LR Scheduler: {scheduler_name}")
        if isinstance(scheduler, ReduceLROnPlateau):
             print(f"  Scheduler Mode: {scheduler.mode}, Factor: {scheduler.factor}, Patience: {scheduler.patience}, Threshold: {scheduler.threshold}")
        # Add more specific prints for other schedulers if needed
    else:
        print("No LR Scheduler used.")

    if accumulation_steps > 1:
        print(f"Gradient Accumulation enabled. Accumulating over {accumulation_steps} steps.")
        actual_batch_size = getattr(train_loader, 'batch_size', 'N/A')
        if isinstance(actual_batch_size, int):
             effective_batch_size = actual_batch_size * accumulation_steps
             print(f"Actual Batch Size: {actual_batch_size}, Effective Batch Size: {effective_batch_size}")
        else:
             print(f"Actual Batch Size: {actual_batch_size} (Attribute not found), Effective steps: {accumulation_steps}")
    else:
        print("Gradient Accumulation disabled (accumulation_steps=1).")

    if clip_grad_value is not None:
        print(f"Gradient Clipping by value enabled: Clipping to [{-clip_grad_value}, {clip_grad_value}].")
    else:
        print("Gradient Clipping disabled.")

    # --- Epoch Loop ---
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{num_epochs} | Current LR: {current_lr:.6f}')
        print('-' * 10)

        # --- Warmup Phase ---
        if epoch < warmup_epochs:
            # Simple linear warmup from near 0 to initial_lr
            warmup_factor = (epoch + 1) / warmup_epochs
            new_lr = initial_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"  Warmup Epoch {epoch+1}/{warmup_epochs}: Adjusted LR to {new_lr:.6f}")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad() # Zero gradients at the start of training phase or after step

        pbar_train = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False, mininterval=1.0) # Adjust mininterval as needed
        for step, (inputs, labels) in enumerate(pbar_train):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            original_loss = loss_criterion(outputs, labels)

            # Scale loss for accumulation
            loss = original_loss / accumulation_steps
            # Backward pass (accumulates gradients)
            loss.backward()

            # Accumulate metrics using original loss
            running_loss += original_loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            # Optimizer step, clipping, and zero grad (conditional)
            if (step + 1) % accumulation_steps == 0:
                # Gradient Clipping (before optimizer step)
                if clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)

                optimizer.step() # Update weights
                optimizer.zero_grad() # Reset gradients for next cycle

            # Update progress bar
            pbar_train.set_postfix(loss=f"{original_loss.item():.4f}", acc=f"{(torch.sum(preds == labels.data).item()/inputs.size(0)):.4f}")

        # Handle final gradient update if epoch size not divisible by accumulation steps
        if len(train_loader) % accumulation_steps != 0:
            print(f"Epoch {epoch+1}: Performing final optimizer step for remaining {len(train_loader) % accumulation_steps} steps' gradients.")
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()

        # Calculate and log epoch training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation', leave=False, mininterval=1.0)
        with torch.no_grad():
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                pbar_val.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(torch.sum(preds == labels.data).item()/inputs.size(0)):.4f}")

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.item())
        print(f'Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}')

        # --- LR Scheduler Step ---
        # ReduceLROnPlateau steps based on validation loss metric
        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
             scheduler.step(epoch_val_loss)
        # Other schedulers like CosineAnnealingLR might step here based on epoch
        elif scheduler and isinstance(scheduler, CosineAnnealingLR):
             # Only step if *not* in warmup phase, as warmup handles LR manually
             if epoch >= warmup_epochs:
                  scheduler.step()


        # --- Early Stopping & Best Model Saving ---
        if epoch_val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...')
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved to {best_model_path}")
            except Exception as e:
                 print(f"Error saving model: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve for {epochs_no_improve} epoch(s). Current best: {best_val_loss:.4f}')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs!')
            early_stop = True
            break # Exit epoch loop

    # --- End of Training ---
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    print(f'Validation Accuracy at Best Loss: {best_val_acc:.4f}')

    # Load best model weights
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        except Exception as e:
            print(f"Error loading best model weights: {e}. Returning model in last state.")
    else:
        print("Warning: Best model file not found. Returning the model in its last state.")

    actual_epochs_run = epoch + 1
    return model, train_losses[:actual_epochs_run], val_losses[:actual_epochs_run]


# ==============================================================================
# main Function (Includes Arg Parsing for New Features)
# ==============================================================================
def main():
    current_dir = Path(__file__).parent
    data_root = current_dir / "data"
    print(f"Looking for data in: {data_root}")
    if not data_root.is_dir():
        print(f"Error: Data directory not found at {data_root}")
        return

    # --- Command line argument parsing ---
    parser = argparse.ArgumentParser(description="Train image classification model with advanced features")

    # Model and Task Selection
    parser.add_argument('--model_name', type=str, default='Newmodel', choices=['Newmodel', 'pretrained_resnet', 'more_unfreeze', 'Attention', 'semi_supervised'], help='Choose model')
    parser.add_argument('--task', type=str, required=True, choices=['A', 'B'], help='Task name (A or B)')

    # Basic Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Actual batch size per step')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs') # Increased default
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate') # Adjusted default
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='Choose optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay parameter') # Adjusted default

    # Loss Function
    parser.add_argument('--loss_function', type=str, default='label_smoothing', choices=['cross_entropy', 'focal_loss', 'label_smoothing'], help='Choose loss function') # Changed default
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25, help='Alpha for Focal Loss')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.15, help='Epsilon for Label Smoothing') # Adjusted default

    # Regularization and Stability
    parser.add_argument('--augmentation_level', type=str, default='medium', choices=['weak', 'medium', 'strong'], help='Data augmentation level')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs)') # Increased default
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--clip_grad_value', type=float, default=None, help='Value for gradient clipping by value (e.g., 0.5). None to disable.')

    # Learning Rate Scheduling & Warmup
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of epochs for linear warmup (0 to disable)') # Added default
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau', 'cosine'], help='Type of learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler (epochs)')
    parser.add_argument('--scheduler_factor', type=float, default=0.2, help='Factor for ReduceLROnPlateau') # Adjusted default

    # Data Loading
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # Model Specific (Pretrained / Attention)
    parser.add_argument('--freeze_until', type=str, default=None, help='Layer name to freeze until for pretrained models')
    parser.add_argument('--backbone_channels', type=int, default=64, 
                        help='Output channels of the initial backbone (stem). '
                            'Impact: Increasing this value makes the initial feature maps richer (more channels), '
                            'potentially capturing more low-level details early on. However, it significantly increases '
                            'the computational cost and parameter count in the early layers and subsequent stages. '
                            'Decreasing it reduces cost but might limit the initial feature representation capacity.') # Default back to 64
    parser.add_argument('--stage_channels', type=int, nargs='+', default=[128, 256, 512, 1024], 
                        help='List of output channels for each stage transition layer. Defines the channel dimension '
                            'entering each main stage of the network. '
                            'Impact: Controls the width (channel capacity) of each stage. Increasing channel counts '
                            'allows the network to potentially learn more complex features at each depth level, '
                            'but drastically increases parameters and computation, risking overfitting on smaller datasets. '
                            'Decreasing channels makes the model lighter but might hinder its ability to learn complex patterns. '
                            'The length of this list determines the number of stages in the network (excluding the backbone).')
    parser.add_argument('--stage_hidden_channels', type=int, nargs='+', default=[64, 128, 256, 512], 
                        help='List of hidden channels within the RELAN blocks for each stage. This controls the internal '
                             'expansion/processing dimension inside the RELAN blocks. '
                             'Impact: Similar to stage_channels, increasing hidden channels increases the capacity and '
                             'complexity within each RELAN block, potentially improving feature learning but adding significant '
                             'computational cost and parameters. It allows for richer feature interactions within the block. '
                             'Decreasing it makes blocks cheaper but potentially less expressive. Must have the same number '
                             'of elements as --stage_channels.')
    parser.add_argument('--num_relan_blocks', type=int, nargs='+', default=[2, 2, 4, 2], 
                        help='List specifying the number of RELAN blocks stacked within each stage. '
                             'Impact: Directly controls the depth of each stage and the overall network depth. Increasing the '
                             'number of blocks allows the network to learn more hierarchical and refined features through '
                             'repeated application of the RELAN structure. However, it increases computation, parameters, '
                             'and the risk of vanishing/exploding gradients (though mitigated by residual connections). '
                             'More blocks generally lead to larger receptive fields deeper in the network. Must have the same '
                             'number of elements as --stage_channels.')
    parser.add_argument('--relan_branches', type=int, default=4, 
                        help='Number of parallel branches concatenated within each RELAN block (includes the initial split path). '
                             'Impact: Controls the "width" or parallelism inside the RELAN block structure. More branches '
                             'allow the block to potentially learn more diverse features in parallel before fusion. '
                             'Increases parameters and computation within the block moderately compared to increasing channels. '
                             'Requires `attn_types` in the code to have `relan_branches - 1` elements.')
    parser.add_argument('--area_attn_heads', type=int, default=8, 
                        help='Number of attention heads in the AreaAttention modules. '
                             'Impact: Splits the channel dimension into multiple heads, allowing the attention mechanism to '
                             'focus on different representation subspaces simultaneously. Increasing heads can potentially '
                             'capture more varied feature relationships but adds slight computational overhead and requires '
                             'the channel dimension (`stage_hidden_channels` in this case) to be divisible by the number of heads. '
                             'Too many heads with small head dimensions might limit the expressiveness of each head.')
    parser.add_argument('--area_attn_split_factor', type=int, default=1, 
                        help='Area split factor for AreaAttention. Defines how many segments the flattened spatial dimension '
                             '(H*W) is split into for local attention calculation. '
                             'Impact: A value of 1 (default) performs global attention across the entire feature map (within the module). '
                             'Increasing this value (e.g., 4) splits the feature map into segments and computes attention locally '
                             'within each segment. This can reduce computational cost (quadratic in sequence length) for large feature maps '
                             'and focus on local patterns. However, it requires H*W to be divisible by this factor and limits '
                             'long-range spatial dependencies captured directly by the attention mechanism. For maximum input size '
                             'flexibility, use 1.')
    parser.add_argument('--cbam_reduction', type=int, default=16, 
                        help='Reduction ratio for the MLP bottleneck in CBAM\'s channel attention module. '
                             'Impact: Controls the complexity and parameter count of the channel attention mechanism. A higher '
                             'reduction factor (e.g., 16) creates a smaller bottleneck, reducing parameters and computation '
                             'but potentially limiting the capacity to model complex channel interdependencies. A lower factor '
                             '(e.g., 4 or 8) increases capacity at the cost of more parameters.')
    parser.add_argument('--cbam_kernel_size', type=int, default=7, choices=[3, 7], 
                        help='Kernel size for the convolutional layer in CBAM\'s spatial attention module. '
                             'Impact: Determines the receptive field for aggregating spatial context to generate the spatial '
                             'attention map. A larger kernel (7x7) captures broader spatial context, potentially better for larger '
                             'objects or features, while a smaller kernel (3x3) focuses on more local context and is computationally cheaper.')
    parser.add_argument('--activation', type=str, default='silu', choices=['relu', 'silu', 'gelu'], 
                        help='Specifies the activation function used throughout the network (in ConvBNAct blocks). '
                             'Impact: Different activation functions have different properties. ReLU is simple and common but can suffer '
                             'from "dying ReLU". SiLU (Swish) and GELU are smoother activations often found in newer architectures '
                             '(like EfficientNet, Vision Transformers) and can sometimes lead to slightly better performance, '
                             'though potentially at a marginal computational cost increase.')

    # Clustering Specific (for semi_supervised)
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN epsilon')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN min_samples')

    args = parser.parse_args()

    # --- Determine Number of Classes ---
    num_classes = -1 # Initialize
    if args.task == 'B':
        try:
            num_classes = len(class_to_idx) if 'class_to_idx' in globals() else -1
        except NameError: pass # Handled below
    else: # Task A
        try:
             num_classes = len(superclass_to_idx) if 'superclass_to_idx' in globals() else -1
        except NameError: pass # Handled below
    if num_classes <= 0:
         print("Warning: Could not determine num_classes from data_loader imports.")
         # Fallback attempt is now integrated into data loading section

    # --- Create DataLoaders ---
    print(f"Creating data loaders with actual batch size: {args.batch_size}, augmentation: {args.augmentation_level}, workers: {args.num_workers}")
    print("IMPORTANT: Ensure data_loader.py is configured for the desired image size (e.g., 256x256).")
    data_loader_kwargs = {
        "data_dir": data_root,
        "batch_size": args.batch_size,
        "augmentation_level": args.augmentation_level,
        "num_workers": args.num_workers,
        # "image_size": 256 # Pass this if your create functions accept it
    }
    train_loader, val_loader, test_loader = None, None, None # Initialize
    try:
        if args.task == 'B':
            train_loader, val_loader, _ = create_data_loaders(**data_loader_kwargs)
            if num_classes <= 0: # Try fallback if needed
                 num_classes = len(train_loader.dataset.class_to_idx)
        else: # Task A
            train_loader, val_loader, test_loader = create_data_loaders2(**data_loader_kwargs)
            if num_classes <= 0: # Fallback for Task A (using imported variable is preferred)
                try:
                     num_classes = len(superclass_to_idx)
                     if num_classes <= 0: raise ValueError()
                except (NameError, ValueError):
                     print("Critical Error: Cannot determine num_classes for Task A. superclass_to_idx missing or invalid in data_loader.py.")
                     return
        print(f"Successfully created data loaders. Determined num_classes: {num_classes}")
    except FileNotFoundError as e:
        print(f"Error: Data directory issue - {e}")
        return
    except AttributeError as e:
         print(f"Error: Problem accessing dataset attributes (e.g., class_to_idx). Check data loader implementation. Details: {e}")
         return
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        traceback.print_exc()
        return
    if num_classes <= 0:
         print("Fatal Error: Failed to determine a valid number of classes.")
         return


    # --- Setup Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of classes: {num_classes}")

    # --- Load Loss Function ---
    loss_kwargs = {}
    if args.loss_function == 'label_smoothing':
        loss_kwargs['smoothing'] = args.label_smoothing_epsilon
        loss_kwargs['num_classes'] = num_classes
    elif args.loss_function == 'focal_loss':
        loss_kwargs['alpha'] = args.focal_loss_alpha
        loss_kwargs['gamma'] = args.focal_loss_gamma
    try:
        loss_criterion = get_loss_criterion(args.loss_function, device=device, **loss_kwargs)
        print(f"Loss function '{args.loss_function}' loaded successfully.")
    except ValueError as e:
        print(f"Error getting loss criterion: {e}")
        return
    except Exception as e:
        print(f"Unexpected error getting loss criterion: {e}")
        traceback.print_exc()
        return

    # --- Initialize Model ---
    model = None
    try:
        activation_map = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}
        act_class = activation_map.get(args.activation.lower())
        if act_class is None: raise ValueError(f"Invalid activation: {args.activation}")

        if args.model_name == 'Attention':
            if not (len(args.stage_channels) == len(args.stage_hidden_channels) == len(args.num_relan_blocks)):
                 raise ValueError("List lengths for stage_channels, stage_hidden_channels, and num_relan_blocks must match!")
            model = DynamicRELANClassifier(num_classes=num_classes, backbone_channels=args.backbone_channels, stage_channels=args.stage_channels, stage_hidden_channels=args.stage_hidden_channels, num_relan_blocks=args.num_relan_blocks, relan_branches=args.relan_branches, area_attn_heads=args.area_attn_heads, area_attn_split_factor=args.area_attn_split_factor, cbam_reduction=args.cbam_reduction, cbam_kernel_size=args.cbam_kernel_size, act=act_class)
        elif args.model_name == 'Newmodel': model = Newmodel(num_classes=num_classes)
        elif args.model_name == 'pretrained_resnet': model = pretrained_resnet_model(num_classes=num_classes)
        elif args.model_name == 'more_unfreeze': model = pretrained_resnet_model_flexible_freeze(num_classes=num_classes, freeze_until=args.freeze_until)
        elif args.model_name == 'semi_supervised': model = ClusterModel(args.eps, args.min_samples)
        else: raise ValueError(f"Unknown model name: {args.model_name}")
        model.to(device)
        print(f"Model '{args.model_name}' initialized successfully.")
    except ValueError as e:
        print(f"Error initializing model: {e}")
        return
    except Exception as e:
        print(f"Unexpected error initializing model: {e}")
        traceback.print_exc()
        return

    # --- Initialize Optimizer ---
    optimizer = None
    try:
        if args.optimizer == 'adam': optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd': optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else: raise ValueError(f"Unknown optimizer: {args.optimizer}")
        print(f"Optimizer '{args.optimizer}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        traceback.print_exc()
        return

    # --- Initialize LR Scheduler ---
    scheduler = None
    try:
        if args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
            print(f"Using ReduceLROnPlateau scheduler.")
        elif args.scheduler == 'cosine':
            t_max_epochs = args.epochs - args.warmup_epochs
            if t_max_epochs <= 0: t_max_epochs = args.epochs # Ensure T_max is positive
            # eta_min ensures LR doesn't drop completely to 0
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max_epochs, eta_min=max(1e-7, args.learning_rate * 0.01))
            print(f"Using CosineAnnealingLR scheduler with T_max={t_max_epochs} epochs (after warmup).")
        elif args.scheduler == 'none':
             print("No LR scheduler selected.")
        else:
             print(f"Warning: Unknown scheduler type '{args.scheduler}'. No scheduler will be used.")
    except Exception as e:
        print(f"Error initializing scheduler: {e}")
        traceback.print_exc()
        # Continue without scheduler if init fails


    # --- Print Final Training Configuration ---
    print("\n----- Training Configuration -----")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"Number of classes: {num_classes}")
    print(f"Actual Batch size: {args.batch_size}")
    if args.accumulation_steps > 1:
        effective_batch_size = args.batch_size * args.accumulation_steps
        print(f"Gradient Accumulation Steps: {args.accumulation_steps}")
        print(f"Effective Batch Size: {effective_batch_size}")
    else:
        print("Gradient Accumulation: Disabled")
    print(f"Max number of epochs: {args.epochs}")
    print(f"Early Stopping Patience: {args.patience}")
    print(f"Initial Learning rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer}")
    if args.optimizer == 'sgd': print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"Loss function: {args.loss_function}")
    if args.loss_function == 'focal_loss': print(f"  Focal Loss Alpha: {args.focal_loss_alpha}\n  Focal Loss Gamma: {args.focal_loss_gamma}")
    elif args.loss_function == 'label_smoothing': print(f"  Label Smoothing Epsilon: {args.label_smoothing_epsilon}")
    print(f"Data Augmentation: {args.augmentation_level}")
    # print(f"Input Image Size (Target): 256x256") # Assuming this is set in data_loader
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"LR Scheduler: {args.scheduler}")
    if args.scheduler == 'plateau': print(f"  Scheduler Patience: {args.scheduler_patience}\n  Scheduler Factor: {args.scheduler_factor}")
    elif args.scheduler == 'cosine': print(f"  Cosine T_max (epochs after warmup): {t_max_epochs if 't_max_epochs' in locals() else 'N/A'}")
    print(f"Gradient Clipping Value: {'Disabled' if args.clip_grad_value is None else args.clip_grad_value}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Device: {device}")
    if args.model_name == 'Attention':
         print("Attention Model Specific Params:")
         print(f"  Backbone Channels: {args.backbone_channels}")
         print(f"  Stage Channels: {args.stage_channels}")
         print(f"  Stage Hidden Channels: {args.stage_hidden_channels}")
         print(f"  Num RELAN Blocks: {args.num_relan_blocks}")
         print(f"  RELAN Branches: {args.relan_branches}")
         print(f"  Area Attention Heads: {args.area_attn_heads}")
         print(f"  Area Attention Split Factor: {args.area_attn_split_factor}")
         print(f"  CBAM Reduction: {args.cbam_reduction}")
         print(f"  CBAM Kernel Size: {args.cbam_kernel_size}")
         print(f"  Activation: {args.activation}")
    print("----------------------------------")

    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)

    # --- Train model ---
    print("----- Start Training -----")
    trained_model = None
    train_losses, val_losses = [], []

    if args.model_name == 'semi_supervised':
        print("Training semi-supervised model... (Requires specific implementation)")
        # Add your semi-supervised training call here if implemented
        # e.g., train_k_model(...)
        return # Assuming semi-supervised is separate for now
    else:
        try:
            trained_model, train_losses, val_losses = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_criterion=loss_criterion,
                optimizer=optimizer,
                scheduler=scheduler, # Pass the scheduler
                num_epochs=args.epochs,
                device=device,
                runs_dir=str(runs_dir),
                patience=args.patience,
                accumulation_steps=args.accumulation_steps,
                warmup_epochs=args.warmup_epochs,
                clip_grad_value=args.clip_grad_value
            )
        except Exception as e:
            print(f"\nAn error occurred during training: {e}")
            traceback.print_exc()
            return # Exit if training fails

    # --- Visualize loss curves ---
    if trained_model and train_losses and val_losses:
        print("Visualizing loss curves...")
        try:
            visualize_loss_curves(train_losses, val_losses, args.model_name, args.loss_function, args.optimizer)
        except Exception as e:
            print(f"Error visualizing loss curves: {e}")
            traceback.print_exc()
    else:
        print("Skipping loss visualization as training did not complete or produce results.")

if __name__ == '__main__':
    # Standard checks at script start
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        try:
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Could not get CUDA device name: {e}")

    main() # Execute the main function

# --- END OF FILE train.py ---