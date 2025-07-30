import numpy as np
import random
import copy

# --- 1. Forward Pass ---
def forward_pass(input_data, weights, biases, hidden_activation_function, output_activation_function, is_training=True, dropout_masks=None, dropout_rate=0.0):
    """
    Calculates the output of the neural network for a given input.
    Includes dropout functionality during training.

    Args:
        input_data (numpy.ndarray): A vector or matrix representing the input features.
        weights (list): A list of weight matrices for each layer.
        biases (list): A list of bias vectors for each layer.
        hidden_activation_function (function): Activation function for hidden layers (e.g., sigmoid, relu).
        output_activation_function (function): Activation function for the output layer (e.g., linear, sigmoid).
        is_training (bool): Flag indicating if the network is in training mode (True) or inference mode (False).
        dropout_masks (list, optional): List to store dropout masks generated during training.
        dropout_rate (float): The probability of dropping a neuron's activation during training.

    Returns:
        tuple: A tuple containing:
            - output_prediction (numpy.ndarray): The final output of the network.
            - activations (list): List of activations for each layer (including input).
            - weighted_sums (list): List of weighted sums (z values) for each layer.
    """
    activations = [input_data]  # Store activations for each layer (including input)
    weighted_sums = []        # Store weighted sums (z values) for each layer

    L = len(weights) # Total number of layers (excluding input)

    for l in range(L): # Loop through layers, 0-indexed for Python lists
        # Calculate the weighted sum for the current layer
        z_l = np.dot(weights[l], activations[l]) + biases[l]
        weighted_sums.append(z_l)

        # Apply the appropriate activation function
        if l < L - 1: # Hidden layers
            a_l = hidden_activation_function(z_l)
        else: # Output layer
            a_l = output_activation_function(z_l)

        # Apply dropout if in training mode and not the output layer
        if is_training and dropout_rate > 0 and l < L - 1: # Dropout typically not on output layer
            mask = (np.random.rand(*a_l.shape) > dropout_rate) / (1 - dropout_rate)
            a_l *= mask
            if dropout_masks is not None:
                dropout_masks.append(mask) # Store mask for backpropagation

        activations.append(a_l)

    output_prediction = activations[L] # The final output of the network (last activation)
    return output_prediction, activations, weighted_sums

# --- 4. Derivative Functions (Examples) ---

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the Sigmoid activation function."""
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    """Linear activation function."""
    return x

def linear_derivative(x):
    """Derivative of the Linear activation function."""
    return np.ones_like(x)

def mean_squared_error(output, target):
    """
    Mean Squared Error loss function for continuous values.
    Args:
        output (numpy.ndarray): Predicted continuous values.
        target (numpy.ndarray): True continuous values.
    Returns:
        float: The mean squared error loss.
    """
    return 0.5 * np.sum((output - target)**2)

def mean_squared_error_derivative(output, target):
    """
    Derivative of the Mean Squared Error loss function with respect to output.
    Args:
        output (numpy.ndarray): Predicted continuous values.
        target (numpy.ndarray): True continuous values.
    Returns:
        numpy.ndarray: The derivative of the MSE loss.
    """
    return output - target

# --- 2. Backpropagation ---
def backpropagation(output_prediction, target_labels, activations, weighted_sums, weights,
                    hidden_activation_function_derivative, output_activation_function_derivative, dropout_masks=None):
    """
    Calculates the gradients of the loss function with respect to the weights and biases.
    Applies dropout masks to error terms during backpropagation.

    Args:
        output_prediction (numpy.ndarray): The output from the forward pass.
        target_labels (numpy.ndarray): The true labels for the input data.
        activations (list): List of activations from the forward pass.
        weighted_sums (list): List of weighted sums (z values) from the forward pass.
        weights (list): List of weight matrices.
        hidden_activation_function_derivative (function): Derivative of the hidden activation function.
        output_activation_function_derivative (function): Derivative of the output activation function.
        loss_function_derivative (function): A function that computes the derivative
                                             of the loss function.
        dropout_masks (list, optional): List of dropout masks generated during the forward pass.
                                        Each mask corresponds to a hidden layer where dropout was applied.

    Returns:
        tuple: A tuple containing:
            - gradient_weights (list): List of gradients for weights.
            - gradient_biases (list): List of gradients for biases.
    """
    num_layers = len(weights)
    delta = [] # Store error terms (delta values) for each layer
    gradient_weights = [None] * num_layers # Initialize with None, will fill in order
    gradient_biases = [None] * num_layers  # Initialize with None, will fill in order

    # 1. Calculate the error for the output layer (L)
    dC_daL = mean_squared_error_derivative(output_prediction, target_labels) # Using MSE derivative for continuous output
    daL_dzL = output_activation_function_derivative(weighted_sums[num_layers-1])
    delta_L = dC_daL * daL_dzL
    delta.append(delta_L) # delta for the last layer

    # 2. Backpropagate the error through the hidden layers (L-1 down to 1)
    dropout_mask_idx = len(dropout_masks) - 1 if dropout_masks else -1

    for l in range(num_layers - 2, -1, -1):
        # Calculate the error term for the current layer
        delta_l = np.dot(weights[l+1].T, delta[0]) * hidden_activation_function_derivative(weighted_sums[l])

        # Apply dropout mask if available for this layer
        if dropout_masks is not None and dropout_mask_idx >= 0:
            delta_l *= dropout_masks[dropout_mask_idx]
            dropout_mask_idx -= 1 # Move to the mask for the previous layer

        delta.insert(0, delta_l) # Insert at the beginning to maintain correct order for layers

    # 3. Calculate gradients for weights and biases
    for l in range(num_layers):
        grad_W_l = np.dot(delta[l], activations[l].T)
        gradient_weights[l] = grad_W_l

        grad_B_l = delta[l]
        gradient_biases[l] = grad_B_l

    return gradient_weights, gradient_biases

# --- 3. Optimizer Functions ---

def gradient_descent_optimizer(weights, biases, gradient_weights, gradient_biases, learning_rate, optimizer_state=None):
    """
    Updates weights and biases using standard Gradient Descent.
    """
    updated_weights = [w - learning_rate * gw for w, gw in zip(weights, gradient_weights)]
    updated_biases = [b - learning_rate * gb for b, gb in zip(biases, gradient_biases)]
    return updated_weights, updated_biases, optimizer_state # optimizer_state is unchanged

def adam_optimizer(weights, biases, gradient_weights, gradient_biases, learning_rate, optimizer_state):
    """
    Updates weights and biases using Adam optimizer.

    Args:
        weights (list): Current weight matrices.
        biases (list): Current bias vectors.
        gradient_weights (list): Gradients for weights.
        gradient_biases (list): Gradients for biases.
        learning_rate (float): Learning rate.
        optimizer_state (dict): Dictionary containing Adam's internal state (m, v, t, beta1, beta2, epsilon).

    Returns:
        tuple: Updated weights, biases, and updated optimizer_state.
    """
    beta1 = optimizer_state.get('beta1', 0.9)
    beta2 = optimizer_state.get('beta2', 0.999)
    epsilon = optimizer_state.get('epsilon', 1e-8)

    t = optimizer_state['t'] + 1
    optimizer_state['t'] = t

    if 'm_w' not in optimizer_state:
        optimizer_state['m_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['v_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['m_b'] = [np.zeros_like(b) for b in biases]
        optimizer_state['v_b'] = [np.zeros_like(b) for b in biases]

    updated_weights = []
    updated_biases = []

    for i in range(len(weights)):
        # Update moments for weights
        optimizer_state['m_w'][i] = beta1 * optimizer_state['m_w'][i] + (1 - beta1) * gradient_weights[i]
        optimizer_state['v_w'][i] = beta2 * optimizer_state['v_w'][i] + (1 - beta2) * (gradient_weights[i]**2)

        # Bias correction
        m_hat_w = optimizer_state['m_w'][i] / (1 - beta1**t)
        v_hat_w = optimizer_state['v_w'][i] / (1 - beta2**t)

        # Update weights
        updated_weights.append(weights[i] - learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon))

        # Update moments for biases
        optimizer_state['m_b'][i] = beta1 * optimizer_state['m_b'][i] + (1 - beta1) * gradient_biases[i]
        optimizer_state['v_b'][i] = beta2 * optimizer_state['v_b'][i] + (1 - beta2) * (gradient_biases[i]**2)

        # Bias correction
        m_hat_b = optimizer_state['m_b'][i] / (1 - beta1**t)
        v_hat_b = optimizer_state['v_b'][i] / (1 - beta2**t)

        # Update biases
        updated_biases.append(biases[i] - learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon))

    return updated_weights, updated_biases, optimizer_state

def adamax_optimizer(weights, biases, gradient_weights, gradient_biases, learning_rate, optimizer_state):
    """
    Updates weights and biases using Adamax optimizer.

    Args:
        weights (list): Current weight matrices.
        biases (list): Current bias vectors.
        gradient_weights (list): Gradients for weights.
        gradient_biases (list): Gradients for biases.
        learning_rate (float): Learning rate.
        optimizer_state (dict): Dictionary containing Adamax's internal state (exp_avg, exp_inf_norm, t, beta1, beta2, epsilon).

    Returns:
        tuple: Updated weights, biases, and updated optimizer_state.
    """
    beta1 = optimizer_state.get('beta1', 0.9)
    beta2 = optimizer_state.get('beta2', 0.999)
    epsilon = optimizer_state.get('epsilon', 1e-8) # Adamax typically doesn't use epsilon in denominator, but good to have

    t = optimizer_state['t'] + 1
    optimizer_state['t'] = t

    if 'exp_avg_w' not in optimizer_state:
        optimizer_state['exp_avg_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['exp_inf_norm_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['exp_avg_b'] = [np.zeros_like(b) for b in biases]
        optimizer_state['exp_inf_norm_b'] = [np.zeros_like(b) for b in biases]

    updated_weights = []
    updated_biases = []

    for i in range(len(weights)):
        # Update exponential average of gradients for weights
        optimizer_state['exp_avg_w'][i] = beta1 * optimizer_state['exp_avg_w'][i] + (1 - beta1) * gradient_weights[i]
        # Update exponentially weighted infinity norm for weights
        optimizer_state['exp_inf_norm_w'][i] = np.maximum(beta2 * optimizer_state['exp_inf_norm_w'][i], np.abs(gradient_weights[i]))

        # Update weights
        updated_weights.append(weights[i] - (learning_rate / (1 - beta1**t)) * optimizer_state['exp_avg_w'][i] / (optimizer_state['exp_inf_norm_w'][i] + epsilon))

        # Update exponential average of gradients for biases
        optimizer_state['exp_avg_b'][i] = beta1 * optimizer_state['exp_avg_b'][i] + (1 - beta1) * gradient_biases[i]
        # Update exponentially weighted infinity norm for biases
        optimizer_state['exp_inf_norm_b'][i] = np.maximum(beta2 * optimizer_state['exp_inf_norm_b'][i], np.abs(gradient_biases[i]))

        # Update biases
        updated_biases.append(biases[i] - (learning_rate / (1 - beta1**t)) * optimizer_state['exp_avg_b'][i] / (optimizer_state['exp_inf_norm_b'][i] + epsilon))

    return updated_weights, updated_biases, optimizer_state

def radam_optimizer(weights, biases, gradient_weights, gradient_biases, learning_rate, optimizer_state):
    """
    Updates weights and biases using RAdam optimizer.
    This is a simplified implementation focusing on the core rectification idea.

    Args:
        weights (list): Current weight matrices.
        biases (list): Current bias vectors.
        gradient_weights (list): Gradients for weights.
        gradient_biases (list): Gradients for biases.
        learning_rate (float): Learning rate.
        optimizer_state (dict): Dictionary containing RAdam's internal state (m, v, t, beta1, beta2, epsilon).

    Returns:
        tuple: Updated weights, biases, and updated optimizer_state.
    """
    beta1 = optimizer_state.get('beta1', 0.9)
    beta2 = optimizer_state.get('beta2', 0.999)
    epsilon = optimizer_state.get('epsilon', 1e-8)

    t = optimizer_state['t'] + 1
    optimizer_state['t'] = t

    if 'm_w' not in optimizer_state:
        optimizer_state['m_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['v_w'] = [np.zeros_like(w) for w in weights]
        optimizer_state['m_b'] = [np.zeros_like(b) for b in biases]
        optimizer_state['v_b'] = [np.zeros_like(b) for b in biases]

    updated_weights = []
    updated_biases = []

    rho_inf = 2 / (1 - beta2) - 1 # Constant for RAdam's variance rectification

    for i in range(len(weights)):
        # Update moments for weights
        optimizer_state['m_w'][i] = beta1 * optimizer_state['m_w'][i] + (1 - beta1) * gradient_weights[i]
        optimizer_state['v_w'][i] = beta2 * optimizer_state['v_w'][i] + (1 - beta2) * (gradient_weights[i]**2)

        # Bias correction for moments
        m_hat_w = optimizer_state['m_w'][i] / (1 - beta1**t)
        v_hat_w = optimizer_state['v_w'][i] / (1 - beta2**t)

        # RAdam specific rectification
        rho_t = rho_inf - 2 * t * (1 - beta2**t) / (1 - beta2)

        if rho_t > 4: # Condition for using rectified variance
            r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            adaptive_lr_w = learning_rate * r_t / (np.sqrt(v_hat_w) + epsilon)
            updated_weights.append(weights[i] - adaptive_lr_w * m_hat_w)
        else: # Fallback to standard SGD-like update if variance is low
            updated_weights.append(weights[i] - learning_rate * m_hat_w) # Or learning_rate * gradient_weights[i] for pure SGD

        # Update moments for biases
        optimizer_state['m_b'][i] = beta1 * optimizer_state['m_b'][i] + (1 - beta1) * gradient_biases[i]
        optimizer_state['v_b'][i] = beta2 * optimizer_state['v_b'][i] + (1 - beta2) * (gradient_biases[i]**2)

        # Bias correction for moments
        m_hat_b = optimizer_state['m_b'][i] / (1 - beta1**t)
        v_hat_b = optimizer_state['v_b'][i] / (1 - beta2**t)

        # RAdam specific rectification for biases
        if rho_t > 4:
            adaptive_lr_b = learning_rate * r_t / (np.sqrt(v_hat_b) + epsilon)
            updated_biases.append(biases[i] - adaptive_lr_b * m_hat_b)
        else:
            updated_biases.append(biases[i] - learning_rate * m_hat_b)

    return updated_weights, updated_biases, optimizer_state


# --- Overall Training Loop ---
def train_neural_network(training_data, target_data, num_epochs, learning_rate, network_architecture,
                         hidden_activation_function, hidden_activation_function_derivative,
                         output_activation_function, output_activation_function_derivative,
                         loss_function, loss_function_derivative,
                         optimizer_name='sgd', optimizer_params=None,
                         validation_data=None, validation_targets=None,
                         patience=None, min_delta=0, dropout_rate=0.0,
                         verbose=True): # Added verbose flag for tuning
    """
    Trains a neural network using forward pass and backpropagation with chosen optimizer.
    Includes optional early stopping based on validation loss and dropout layers.

    Args:
        training_data (list): List of input samples for training.
        target_data (list): List of target labels for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for gradient descent.
        network_architecture (list): List of integers defining the number of neurons
                                     in each layer (e.g., [input_dim, hidden1, output_dim]).
        hidden_activation_function (function): Activation function for hidden layers.
        hidden_activation_function_derivative (function): Derivative of hidden activation function.
        output_activation_function (function): Activation function for the output layer.
        output_activation_function_derivative (function): Derivative of output activation function.
        loss_function (function): The loss function to use.
        loss_function_derivative (function): The derivative of the loss function.
        optimizer_name (str): Name of the optimizer to use ('sgd', 'adam', 'adamax', 'radam').
        optimizer_params (dict, optional): Dictionary of parameters for the optimizer.
        validation_data (list, optional): List of input samples for validation.
        validation_targets (list, optional): List of target labels for validation.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped.
        min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement.
        dropout_rate (float): The probability of dropping a neuron's activation during training.
                              Applied to hidden layers only.
        verbose (bool): If True, print training progress.

    Returns:
        tuple: A tuple containing:
            - weights (list): The trained weight matrices.
            - biases (list): The trained bias vectors.
            - final_val_loss (float): The validation loss at the end of training (or best if early stopped).
    """
    # Initialize weights and biases randomly
    weights = []
    biases = []
    for i in range(len(network_architecture) - 1):
        weights.append(np.random.randn(network_architecture[i+1], network_architecture[i]) * 0.01)
        biases.append(np.random.randn(network_architecture[i+1], 1) * 0.01)

    # Convert training data to numpy arrays for consistency
    training_data_np = [np.array(x).reshape(-1, 1) for x in training_data]
    target_data_np = [np.array(y).reshape(-1, 1) for y in target_data]

    # Convert validation data to numpy arrays if provided
    validation_data_np = None
    validation_targets_np = None
    if validation_data is not None and validation_targets is not None:
        validation_data_np = [np.array(x).reshape(-1, 1) for x in validation_data]
        validation_targets_np = [np.array(y).reshape(-1, 1) for y in validation_targets]

    # Initialize optimizer state
    optimizer_state = {'t': 0}
    if optimizer_params:
        optimizer_state.update(optimizer_params)

    optimizers = {
        'sgd': gradient_descent_optimizer,
        'adam': adam_optimizer,
        'adamax': adamax_optimizer,
        'radam': radam_optimizer,
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized. Choose from {list(optimizers.keys())}")

    current_optimizer = optimizers[optimizer_name]

    if verbose:
        print(f"Training with {optimizer_name.upper()} optimizer (Dropout Rate: {dropout_rate})...")

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = [w.copy() for w in weights] # Store initial best weights
    best_biases = [b.copy() for b in biases]   # Store initial best biases
    final_val_loss = float('inf') # Default in case no validation data or early stopping

    for epoch in range(num_epochs):
        total_train_loss = 0
        for i in range(len(training_data_np)):
            input_sample = training_data_np[i]
            target_label = target_data_np[i]

            dropout_masks = [] # List to store masks for current forward pass

            # Forward Pass (training mode)
            output_prediction, activations, weighted_sums = forward_pass(
                input_sample, weights, biases, hidden_activation_function, output_activation_function,
                is_training=True, dropout_masks=dropout_masks, dropout_rate=dropout_rate
            )

            # Backpropagation (with dropout masks)
            gradient_weights, gradient_biases = backpropagation(
                output_prediction, target_label, activations, weighted_sums,
                weights, hidden_activation_function_derivative, output_activation_function_derivative,
                dropout_masks=dropout_masks
            )

            # Update Parameters using the chosen optimizer
            weights, biases, optimizer_state = current_optimizer(
                weights, biases, gradient_weights, gradient_biases, learning_rate, optimizer_state
            )

            total_train_loss += loss_function(output_prediction, target_label)

        avg_train_loss = total_train_loss / len(training_data_np)

        # Early Stopping Check
        if validation_data_np is not None and patience is not None:
            total_val_loss = 0
            for i in range(len(validation_data_np)):
                val_input = validation_data_np[i]
                val_target = validation_targets_np[i]
                # Forward Pass for validation (inference mode - no dropout)
                val_prediction, _, _ = forward_pass(val_input, weights, biases, hidden_activation_function, output_activation_function, is_training=False)
                total_val_loss += loss_function(val_prediction, val_target)
            avg_val_loss = total_val_loss / len(validation_data_np)
            final_val_loss = avg_val_loss # Update final_val_loss with current validation loss

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_weights = [w.copy() for w in weights]
                best_biases = [b.copy() for b in biases]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
                weights = best_weights # Restore best weights
                biases = best_biases   # Restore best biases
                final_val_loss = best_val_loss # Set final_val_loss to the best observed
                break # Exit training loop
        else:
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    return weights, biases, final_val_loss


def tune_hyperparameters(training_data, target_data, validation_data, validation_targets,
                         input_dim, output_dim, num_trials=10):
    """
    Performs a simple random search for hyperparameter tuning.

    Args:
        training_data (list): List of input samples for training.
        target_data (list): List of target labels for training.
        validation_data (list): List of input samples for validation.
        validation_targets (list): List of target labels for validation.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output predictions.
        num_trials (int): Number of random hyperparameter combinations to try.

    Returns:
        dict: A dictionary containing the best hyperparameters found.
    """
    # Define the search space for hyperparameters
    search_space = {
        'num_hidden_layers': [1, 2, 3],
        'max_nodes_per_hidden_layer': [5, 10, 20, 30, 50], # Increased max nodes
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4], # Added more dropout options
        'optimizer_name': ['sgd', 'adam', 'adamax', 'radam'],
        'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.05], # Finer grain and lower values
        'num_epochs': [10000, 20000, 30000, 40000, 50000], # Increased max epochs
        'patience': [500, 1000, 2000], # Increased patience options
        'min_delta': [1e-4, 1e-5, 1e-6] # Finer min_delta
    }

    best_val_loss = float('inf')
    best_hyperparams = {}
    best_model_weights = None
    best_model_biases = None

    print(f"\n--- Starting Hyperparameter Tuning for {num_trials} trials ---")

    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")

        # Randomly select hyperparameters for this trial
        num_hidden_layers = random.choice(search_space['num_hidden_layers'])
        max_nodes_per_hidden_layer = random.choice(search_space['max_nodes_per_hidden_layer'])
        dropout_rate = random.choice(search_space['dropout_rate'])
        optimizer_name = random.choice(search_space['optimizer_name'])
        learning_rate = random.choice(search_space['learning_rate'])
        num_epochs = random.choice(search_space['num_epochs'])
        patience = random.choice(search_space['patience'])
        min_delta = random.choice(search_space['min_delta'])

        # Construct network architecture based on chosen hidden layers and nodes
        network_architecture = [input_dim]
        for _ in range(num_hidden_layers):
            network_architecture.append(max_nodes_per_hidden_layer)
        network_architecture.append(output_dim)

        # Optimizer specific parameters (fixed for now, could also be tuned)
        optimizer_params = {}
        if optimizer_name in ['adam', 'adamax', 'radam']:
            optimizer_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}

        current_hyperparams = {
            'num_hidden_layers': num_hidden_layers,
            'max_nodes_per_hidden_layer': max_nodes_per_hidden_layer,
            'dropout_rate': dropout_rate,
            'optimizer_name': optimizer_name,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'patience': patience,
            'min_delta': min_delta,
            'network_architecture': network_architecture # Store for reference
        }
        print(f"Current Hyperparameters: {current_hyperparams}")

        # Train the model with current hyperparameters
        current_weights, current_biases, current_val_loss = train_neural_network(
            training_data,
            target_data,
            num_epochs,
            learning_rate,
            network_architecture,
            sigmoid, # Hidden activation
            sigmoid_derivative, # Hidden activation derivative
            linear, # Output activation
            linear_derivative, # Output activation derivative
            mean_squared_error,
            mean_squared_error_derivative,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            validation_data=validation_data,
            validation_targets=validation_targets,
            patience=patience,
            min_delta=min_delta,
            dropout_rate=dropout_rate,
            verbose=False # Suppress verbose output during tuning trials
        )

        print(f"Trial {trial + 1} finished with Validation Loss: {current_val_loss:.4f}")

        # Check if this is the best model so far
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_hyperparams = current_hyperparams
            best_model_weights = copy.deepcopy(current_weights)
            best_model_biases = copy.deepcopy(current_biases)
            print(f"New best validation loss: {best_val_loss:.4f}")

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Hyperparameters: {best_hyperparams}")

    return best_hyperparams, best_model_weights, best_model_biases


# --- Example Usage ---
if __name__ == "__main__":
    # Define input and output dimensions for the synthetic dataset
    input_dim = 2
    output_dim = 1

    # --- Generate a synthetic dataset for continuous values ---
    def generate_synthetic_data(num_samples=1000):
        X = np.random.rand(num_samples, input_dim) * 10 - 5 # Features between -5 and 5
        # Example function: y = x1 * sin(x2) + cos(x1) + noise
        y = (X[:, 0] * np.sin(X[:, 1]) + np.cos(X[:, 0]) + np.random.randn(num_samples) * 0.1).reshape(-1, 1)
        return X, y

    num_samples = 500 # Reduced for faster execution in tuning example
    X_data, y_data = generate_synthetic_data(num_samples)

    # Split data into training and validation sets
    split_ratio = 0.8
    split_idx = int(num_samples * split_ratio)

    training_inputs_raw = X_data[:split_idx].tolist()
    training_targets_raw = y_data[:split_idx].tolist()
    validation_inputs_raw = X_data[split_idx:].tolist()
    validation_targets_raw = y_data[split_idx:].tolist()

    # --- Run a single training with default parameters (for comparison) ---
    print("--- Running a single training with default parameters ---")
    # Default network architecture for a 2-input, 1-output problem with 1 hidden layer of 10 nodes
    network_architecture_default = [input_dim, 10, output_dim]
    epochs_default = 10000
    lr_default = 0.005
    patience_default = 500
    min_delta_default = 1e-5
    dropout_rate_default = 0.0

    trained_weights_default, trained_biases_default, final_val_loss_default = train_neural_network(
        training_inputs_raw,
        training_targets_raw,
        epochs_default,
        lr_default,
        network_architecture_default,
        sigmoid, # Hidden activation
        sigmoid_derivative, # Hidden activation derivative
        linear, # Output activation
        linear_derivative, # Output activation derivative
        mean_squared_error, # MSE for output layer
        mean_squared_error_derivative,
        optimizer_name='adam', # Using Adam as a common default
        optimizer_params={'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
        validation_data=validation_inputs_raw,
        validation_targets=validation_targets_raw,
        patience=patience_default,
        min_delta=min_delta_default,
        dropout_rate=dropout_rate_default,
        verbose=True
    )
    print(f"\nDefault Training Final Validation Loss: {final_val_loss_default:.4f}")
    print("\nTesting default trained network (first 5 validation samples):")
    for i in range(min(5, len(validation_inputs_raw))):
        input_np = np.array(validation_inputs_raw[i]).reshape(-1, 1)
        prediction, _, _ = forward_pass(input_np, trained_weights_default, trained_biases_default, sigmoid, linear, is_training=False)
        print(f"Input: {validation_inputs_raw[i]}, Target: {validation_targets_raw[i][0]:.4f}, Prediction: {prediction[0][0]:.4f}")


    # --- Perform Hyperparameter Tuning ---
    num_tuning_trials = 5 # Set a small number of trials for quick demonstration. Increase for better results.

    best_hyperparams_found, best_tuned_weights, best_tuned_biases = tune_hyperparameters(
        training_inputs_raw,
        training_targets_raw,
        validation_inputs_raw,
        validation_targets_raw,
        input_dim,
        output_dim,
        num_trials=num_tuning_trials
    )

    print("\n--- Testing Best Tuned Network (first 5 validation samples) ---")
    if best_tuned_weights and best_tuned_biases:
        for i in range(min(5, len(validation_inputs_raw))):
            input_np = np.array(validation_inputs_raw[i]).reshape(-1, 1)
            prediction, _, _ = forward_pass(input_np, best_tuned_weights, best_tuned_biases, sigmoid, linear, is_training=False)
            print(f"Input: {validation_inputs_raw[i]}, Target: {validation_targets_raw[i][0]:.4f}, Prediction: {prediction[0][0]:.4f}")
    else:
        print("No best model found during tuning (perhaps due to very few trials or issues).")


