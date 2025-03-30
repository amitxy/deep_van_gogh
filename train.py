import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb
import os
import utils

def early_stop_check(patience,
                     best_value,
                     best_value_epoch,
                     current_value,
                     current_value_epoch,
                     current_metrics,
                     best_metrics,
                     direction='maximize'):
    """
    Checks whether training should stop early to prevent overfitting or excessive computation.
    This function compares the current validation metric with the best recorded validation metric.
    If no improvement is observed within the allowed patience (number of epochs), training stops early.
    """
    early_stop_flag = False
    direction = 1 if direction == 'maximize' else -1
    if (current_value - best_value) * direction > 0:
        # Update the parameters holding the best validation loss details
        best_value = current_value
        best_value_epoch = current_value_epoch
        best_metrics = current_metrics.copy()
    else:
        # Check if more than acceptable epochs have passed without improvement
        if current_value_epoch - best_value_epoch > patience:
            early_stop_flag = True
    return best_value, best_value_epoch, early_stop_flag, best_metrics


def train_model_with_hyperparams(model,
                                 train_loader,
                                 val_loader,
                                 optimizer,
                                 criterion,
                                 epochs,
                                 patience,
                                 device,
                                 trial=None,
                                 fold=None,
                                 architecture='model',
                                 save_model=False,
                                 log=True):
    best_value = float('-inf')  # Initialize the best validation loss, -inf because we maximize
    best_value_epoch = 0  # Track epoch with the best validation loss
    early_stop_flag = False
    best_model_state = None  # To save the best model
    best_metrics = None

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()  # Enable training mode
        train_loss = 0.0 # Initializing the cumulative training loss for the current epoch to 0.
        total_train_samples = 0 # Initializing the counter for the total number of training samples processed in the current epoch.
        correct_train_predictions = 0 # Initializing the counter for the total number of correctly predicted training samples.

        for inputs, labels in train_loader: #Iterates over the train_loader, which is a DataLoader object containing batches of training data. Each iteration yields a batch of inputs (images) and corresponding labels (ground-truth classes).
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights using the optimizer

            # Accumulate training loss
            train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

            # Calculate correct predictions for training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train_predictions += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss /= total_train_samples
        train_accuracy = correct_train_predictions / total_train_samples

        val_metrics = validation(model, criterion, val_loader, device)
        val_auc = val_metrics['Validation AUC']

        # Check for early stopping
        best_value, best_value_epoch, early_stop_flag, best_metrics = early_stop_check(patience,
                                                                         best_value,
                                                                         best_value_epoch,
                                                                         val_auc,
                                                                         epoch,
                                                                         val_metrics,
                                                                         best_metrics,
                                                                         direction='maximize')

        # Save the best model under the best_model_state parameter and it's optimizer
        if val_auc == best_value:
            best_model_state = model.state_dict()

        # Log metrics to Weights & Biases - THIS IS WHERE WE TRACK THE RESULTS AND THE PROCESS
        if log:
            track = {"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_accuracy}
            track.update(val_metrics)
            wandb.log(track)

        if early_stop_flag: # Checks whether the early stopping condition has been met, as indicated by the early_stop_flag
            break # Exits the training loop immediately if the early stopping condition is satisfied

    if log or save_model:
        state_dict = best_model_state if (save_model and best_model_state) else None
        best_metrics['Best Epoch'] = best_value_epoch
        save_checkpoint(architecture=architecture,
                    model_state_dict=state_dict,
                    optimizer=optimizer,
                    epochs=epochs,
                    trial=trial,
                    fold=fold,
                    best_metrics=best_metrics)
    return best_metrics

def validation(model, criterion, val_loader, device, is_test=False):
    # Validation loop
    model.eval()  # Enable evaluation mode
    val_loss = 0.0  # Same initialization as in the train
    total_val_samples = 0  # Same initialization as in the train
    correct_val_predictions = 0  # Same initialization as in the train

    # For AUC calculation - pre-allocate arrays
    all_val_labels = torch.zeros(len(val_loader.dataset), dtype=torch.long)
    all_val_probs = torch.zeros(len(val_loader.dataset), dtype=torch.float32)
    all_val_preds = torch.zeros(len(val_loader.dataset), dtype=torch.float32)
    idx = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in val_loader:  # iterate on the val_loader's batches
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            total_val_samples += inputs.size(0)

            # Calculate correct predictions for validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val_predictions += (predicted == labels).sum().item()
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)[:, 1]

            # Store in pre-allocated arrays
            batch_size = labels.size(0)
            all_val_labels[idx:idx + batch_size] = labels.cpu()
            all_val_probs[idx:idx + batch_size] = probs.cpu()
            all_val_preds[idx:idx + batch_size] = predicted.cpu()
            idx += batch_size

    # Calculate validation matrices that we want to track
    val_loss /= total_val_samples
    val_accuracy = correct_val_predictions / total_val_samples
    val_auc = roc_auc_score(all_val_labels.numpy(), all_val_probs.numpy())
    val_F1 = f1_score(all_val_labels.numpy(), all_val_preds.numpy(), average='weighted')
    val_precision = precision_score(all_val_labels.numpy(), all_val_preds.numpy(), average='weighted')
    val_recall = recall_score(all_val_labels.numpy(), all_val_preds.numpy(), average='weighted')
    tn, fp, fn, tp = confusion_matrix(all_val_labels.numpy(), all_val_preds).ravel()
    val_specificity = tn / (tn + fp)
    # Create a wandb confusion matrix plot
    wandb_cm = wandb.plot.confusion_matrix(
        y_true=all_val_labels.numpy(),
        preds=all_val_preds.numpy(),
        class_names=["Not Van Gogh", "Van Gogh"]
    )
    val_type = 'Test' if is_test else 'Validation'
    return {
                f"{val_type} Loss": val_loss,
                f"{val_type} Accuracy": val_accuracy,
                f'{val_type} AUC': val_auc,
                f'{val_type} F1': val_F1,
                f'{val_type} Precision': val_precision,
                f'{val_type} Recall': val_recall,
                f'{val_type} Specificity': val_specificity,
                'Confusion_matrix': wandb_cm
            }

def save_checkpoint(**kwargs):
    architecture = kwargs.get('architecture', 'model')
    best_metrics =  kwargs.get('best_metrics', None)
    if isinstance(best_metrics, dict) and 'Confusion_matrix' in best_metrics:
        metrics = best_metrics.copy()
        metrics.pop('Confusion_matrix', None)  # Remove the confusion matrix from the metrics
        best_metrics = metrics

    save_dir = os.path.join(utils.MODELS_DIR, architecture)
    os.makedirs(save_dir, exist_ok=True)  # Ensures that dir exists


    # Save the model state dict and hyperparameters
    checkpoint = {
        'model_state_dict': kwargs.get('model_state_dict', None),
        'hyperparameters': {
            'architecture': architecture,
            'epochs': kwargs.get('epochs', -1),
            'patience': kwargs.get('patience', -1),
        },
        'best_metrics': best_metrics
    }
    optimizer = kwargs.get('optimizer', None)
    if optimizer:
        checkpoint['hyperparameters'].update({
            'optimizer': optimizer.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
        })

    trial = kwargs.get('trial', None)
    trial_ext = f'_trial_{trial.number + 1}_fold_{kwargs.get("fold", -1)}' if trial else ''
    torch.save(checkpoint, f'{save_dir}/{architecture}_best_model{trial_ext}.pt')  # Save into the same directory