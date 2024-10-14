import json
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_results_to_pdf(resultsfile='../output/vae_models/dev.json', 
                        outputfile='../output/vae_models/dev.pdf'):
    """
    Plot the training and validation losses and metric from a JSON file.
    
    Parameters
    ----------
    resultsfile : str
        Path to the JSON file containing the results.
    """

    # Load the JSON data from the file
    with open(resultsfile, 'r') as f:
        data = json.load(f)

    # Initialize lists to store the values for each epoch
    epochs = []
    train_kl_loss = []
    train_reconstruction_losses = {}
    train_survival_loss = []

    valid_kl_loss = []
    valid_reconstruction_losses = {}
    valid_survival_loss = []
    valid_metric = []

    # Extract the values for each epoch
    for epoch in range(data['params']['epochs']):
        epochs.append(int(epoch))
        epoch = str(epoch)
        
        train_kl_loss.append(data['history'][epoch]['train']['kl_loss'])
        train_survival_loss.append(data['history'][epoch]['train']['survival_loss'])
        
        valid_kl_loss.append(data['history'][epoch]['valid']['kl_loss'])
        valid_survival_loss.append(data['history'][epoch]['valid']['survival_loss'])
        valid_metric.append(data['history'][epoch]['valid']['metric'])
        
        for key, loss in data['history'][epoch]['train']['reconstruction_loss'].items():
            if key not in train_reconstruction_losses:
                train_reconstruction_losses[key] = []
            train_reconstruction_losses[key].append(loss)
        
        for key, loss in data['history'][epoch]['valid']['reconstruction_loss'].items():
            if key not in valid_reconstruction_losses:
                valid_reconstruction_losses[key] = []
            valid_reconstruction_losses[key].append(loss)

    # Sort epochs and corresponding values
    sorted_indices = sorted(range(len(epochs)), key=lambda k: epochs[k])
    epochs = [epochs[i] for i in sorted_indices]
    train_kl_loss = [train_kl_loss[i] for i in sorted_indices]
    train_survival_loss = [train_survival_loss[i] for i in sorted_indices]
    valid_kl_loss = [valid_kl_loss[i] for i in sorted_indices]
    valid_survival_loss = [valid_survival_loss[i] for i in sorted_indices]
    valid_metric = [valid_metric[i] for i in sorted_indices]

    # key here refers to the input type
    for key in train_reconstruction_losses:
        train_reconstruction_losses[key] = [train_reconstruction_losses[key][i] for i in sorted_indices]

    for key in valid_reconstruction_losses:
        valid_reconstruction_losses[key] = [valid_reconstruction_losses[key][i] for i in sorted_indices]

    # Plot the values
    plt.figure(figsize=(12, 8))

    # Calculate the number of subplots needed
    num_train_plots = 2 + len(train_reconstruction_losses)  # KL Loss, Survival Loss, and each reconstruction loss
    num_valid_plots = 3 + len(valid_reconstruction_losses)  # KL Loss, Survival Loss, Metric, and each reconstruction loss

    # Determine the number of rows and columns for the subplots
    num_cols = 2
    num_rows_train = math.ceil(num_train_plots / num_cols)
    num_rows_valid = math.ceil(num_valid_plots / num_cols)

    # Create a PDF file to save the plots
    with PdfPages(outputfile) as pdf:
        # Plot validation losses and metric
        plt.figure(figsize=(12, 8))
        for i, (label, loss) in enumerate([('Valid KL Loss', valid_kl_loss)] + 
                                        [(f'Valid Reconstruction Loss ({key})', loss) for key, loss in valid_reconstruction_losses.items()] + 
                                        [('Valid Survival Loss', valid_survival_loss), ('Valid Metric', valid_metric)]):
            plt.subplot(num_rows_valid, num_cols, i + 1)
            plt.plot(epochs, loss, label=label)
            plt.xlabel('Epoch')
            plt.ylabel('Loss / Metric')
            plt.title(label)
            plt.legend()
        plt.tight_layout()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()
        
        # Plot training losses
        plt.figure(figsize=(12, 8))
        for i, (label, loss) in enumerate([('Train KL Loss', train_kl_loss)] + 
                                        [(f'Train Reconstruction Loss ({key})', loss) for key, loss in train_reconstruction_losses.items()] + 
                                        [('Train Survival Loss', train_survival_loss)]):
            plt.subplot(num_rows_train, num_cols, i + 1)
            plt.plot(epochs, loss, label=label)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(label)
            plt.legend()
        plt.tight_layout()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()