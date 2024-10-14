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

    # matched train and validation losses
    losses = [
        ('KL Loss', train_kl_loss, valid_kl_loss),
        ('Survival Loss', train_survival_loss, valid_survival_loss)
    ] + [
        (f'Reconstruction Loss ({key})', train_reconstruction_losses[key], valid_reconstruction_losses[key])
        for key in train_reconstruction_losses
    ]
    
    # Determine the number of rows and columns for the subplots
    num_cols = 2
    num_rows = math.ceil((3 + len(valid_reconstruction_losses)) / num_cols)

    # Create a PDF file to save the plots
    with PdfPages(outputfile) as pdf:
        
        plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
        
        # Plot validation metric first
        ax1 = plt.subplot(num_rows, num_cols, 1)
        ax1.plot(epochs, valid_metric, 'g-', label='Valid Metric')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metric')
        ax1.set_title('Validation Metric')
        ax1.legend(loc='upper left')
        # then plot the train and valid loss curves
        for i, (label, train_loss, valid_loss) in enumerate(losses):
            ax1 = plt.subplot(num_rows, num_cols, i + 2)
            ax1.plot(epochs, train_loss, 'b-', label=f'Train {label}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Loss')
            ax1.set_title(label)
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.plot(epochs, valid_loss, 'r-', label=f'Valid {label}')
            ax2.set_ylabel('Valid Loss')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()