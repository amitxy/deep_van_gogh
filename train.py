import torch
import wandb

# Checks whether training should stop early to prevent overfitting or excessive computation.
# This function compares the current validation loss with the best recorded validation loss. If no improvement is observed within the allowed patience (number of epochs), it signals that training should stop early.
def early_stop_check(patience, best_val_loss, best_val_loss_epoch, current_val_loss, current_val_loss_epoch):
    early_stop_flag = False  # Initialize flag to be False
    if current_val_loss < best_val_loss:
        # Update the parameters holding the best validation loss details
        best_val_loss = current_val_loss
        best_val_loss_epoch = current_val_loss_epoch
    else:
        # Check if more than acceptable epochs have passed without improvement
        if current_val_loss_epoch - best_val_loss_epoch > patience:
            early_stop_flag = True  # Change flag
    return best_val_loss, best_val_loss_epoch, early_stop_flag


def train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs, patience, trial, device):
    best_val_loss = float('inf')  # Initialize the best validation loss
    best_val_loss_epoch = 0  # Track epoch with the best validation loss
    early_stop_flag = False
    best_model_state = None  # To save the best model in each trial, we could have defined it to same the model after each epoch, save only the best one, and so on...

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

        # Calculate average validation loss and accuracy
        val_loss /= total_val_samples
        val_accuracy = correct_val_predictions / total_val_samples

        # Check for early stopping
        best_val_loss, best_val_loss_epoch, early_stop_flag = early_stop_check(patience, best_val_loss, best_val_loss_epoch, val_loss, epoch)

        # Save the best model under the best_model_state parameter
        if val_loss == best_val_loss:
            best_model_state = model.state_dict()

        # Log metrics to Weights & Biases - THIS IS WHERE WE TRACK THE RESULTS AND THE PROCESS
        wandb.log({ #log == logging of the training process (e.g. results)
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy
        })

        if early_stop_flag: # Checks whether the early stopping condition has been met, as indicated by the early_stop_flag
            break # Exits the training loop immediately if the early stopping condition is satisfied

    # Save the best model as a .pt file
    if best_model_state is not None: # basically just makes sure that there is a better model (if there is an error the val loss will remain -inf)
        torch.save(best_model_state, f"best_model_trial_{trial.number}.pt") # Save into the same directory

    return best_val_loss
