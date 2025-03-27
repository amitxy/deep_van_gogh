import torch


import wandb
import os
import utils
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay


# Checks whether training should stop early to prevent overfitting or excessive computation.
# This function compares the current validation metric with the best recorded validation metric. If no improvement is observed within the allowed patience (number of epochs), it signals that training should stop early.
def early_stop_check(patience, best_value, best_value_epoch, current_value, current_value_epoch, direction='maximize'):
    early_stop_flag = False  # Initialize flag to be False


    if current_value > best_value:
        # Update the parameters holding the best validation loss details
        best_value = current_value
        best_value_epoch = current_value_epoch
    else:
        # Check if more than acceptable epochs have passed without improvement
        if current_value_epoch - best_value_epoch > patience:
            early_stop_flag = True  # Change flag
    return best_value, best_value_epoch, early_stop_flag



def train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs, patience, device, trial, architecture, fold, save_model=False):
    best_value = float('-inf')  # Initialize the best validation loss
    best_value_epoch = 0  # Track epoch with the best validation loss
    early_stop_flag = False

    # To save the best model in each trial
    best_model_state = None
    best_model_optimizer_state = None

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

        # Validation loop
        model.eval()  # Enable evaluation mode
        val_loss = 0.0 # Same initialization as in the train
        total_val_samples = 0 # Same initialization as in the train
        correct_val_predictions = 0 # Same initialization as in the train

        # For AUC calculation - pre-allocate arrays
        all_val_labels = torch.zeros(len(val_loader.dataset), dtype=torch.long)
        all_val_probs = torch.zeros(len(val_loader.dataset), dtype=torch.float32)
        all_val_preds = torch.zeros(len(val_loader.dataset), dtype=torch.float32)
        idx = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader: # iterate on the val_loader's batches
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

        # Calculate average validation loss and accuracy
        val_loss /= total_val_samples
        val_accuracy = correct_val_predictions / total_val_samples

        val_auc = roc_auc_score(all_val_labels.numpy(), all_val_probs.numpy())

        val_F1 = f1_score(all_val_labels.numpy(), all_val_preds.numpy(),average='weighted')

        val_precision = precision_score(all_val_labels.numpy(), all_val_preds.numpy(),average='weighted')

        val_recall = recall_score(all_val_labels.numpy(), all_val_preds.numpy(),average='weighted')

        tn, fp, fn, tp = confusion_matrix(all_val_labels.numpy(), all_val_preds).ravel()
        val_specificity = tn / (tn + fp)

        # Check for early stopping
        best_value, best_value_epoch, early_stop_flag = early_stop_check(patience,
                                                                         best_value,
                                                                         best_value_epoch,
                                                                         val_auc,
                                                                         epoch,
                                                                         direction='maximize')

        # Save the best model under the best_model_state parameter and it's optimizer
        if val_auc == best_value:
            best_model_state = model.state_dict()
            best_model_optimizer_state = optimizer.state_dict()

        if trial is not None:
            # Log metrics to Weights & Biases - THIS IS WHERE WE TRACK THE RESULTS AND THE PROCESS
            wandb.log({ #log == logging of the training process (e.g. results)
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy,
                'Validation AUC': val_auc,
                'Validation F1': val_F1,
                'Validation Precision': val_precision,
                'Validation Recall': val_recall,
                'Validation Specificity': val_specificity
            })

        if early_stop_flag: # Checks whether the early stopping condition has been met, as indicated by the early_stop_flag
            break # Exits the training loop immediately if the early stopping condition is satisfied

    save_dir = os.path.join(utils.MODELS_DIR, architecture)
    os.makedirs(save_dir, exist_ok=True)  # Ensures that dir exists

    # Save the best model as a .pt file
    if best_model_state is not None and trial is not None: # basically just makes sure that there is a better model (if there is an error the val loss will remain -inf)
        torch.save({'optimizer_state_dict': best_model_optimizer_state},
                   f"{save_dir}/best_model_trial_{trial.number}_fold_{fold}.pt") # Save into the same directory

    if save_model:
        torch.save(best_model_state, f'best_model.pt')  # Save into the same directory


    return best_value
