import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from torch.utils.data.dataset import Dataset
import torch
import copy
import os
import pretty_midi
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
from datetime import datetime
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tempfile
import shutil
import contextlib
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import json






#--------------------------#
#     Data Loder           #
#--------------------------#
def sample(y, offset, sample_inp_size, sample_out_size):
    Xin = np.arange(offset, offset + sample_inp_size)
    Xout = np.arange(sample_inp_size + offset, offset + sample_inp_size + sample_out_size)
    out = y[Xout]
    inp = y[Xin]
    return inp, out


def create_dataset(series, n_samples=None, sample_inp_size=51, sample_out_size=1, test=None, verbose=False, plot=False):
    if n_samples is None:
        n_samples = len(series)
    data_inp = np.zeros((n_samples, sample_inp_size))
    data_out = np.zeros((n_samples, sample_out_size))

    for i in range(n_samples):
        sample_inp, sample_out = sample(series, i, sample_inp_size, sample_out_size)
        data_inp[i, :] = sample_inp
        data_out[i, :] = sample_out
    if test is not None:
        assert 0 < test < 1
        split = int(n_samples * (1 - test))
        train_inp, train_out = data_inp[:split], data_out[:split]
        test_inp, test_out = data_inp[split:], data_out[split:]
        series_train = series[:split]
        series_test = series[split:]
    else:
        train_inp, train_out = data_inp, data_out
        test_inp, test_out = data_inp, data_out
        series_train = series
        series_test = series

    dataset_train = LocalDataset(x=train_inp, y=train_out)
    dataset_test = LocalDataset(x=test_inp, y=test_out)

    if verbose:
        print("Train set size: ", dataset_train.length)
        print("Test set size: ", dataset_test.length)

    if plot:
        # Plot generated process.
        plt.plot(np.array(series)[:200])
        plt.show()
    return dataset_train, dataset_test, series_train, series_test


class LocalDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE or L1 Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def generate_armaprocess_data(samples, noise_std, random_order=None, params=None, limit_abs_sum=True):
    if params is not None:
        # use specified params, make sure to sum up to 1 or less
        arparams, maparams = params
        arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=samples)
    else:
        is_stationary = False
        iteration = 0
        while not is_stationary:
            iteration += 1
            # print("Iteration", iteration)
            if iteration > 100:
                raise RuntimeError("failed to find stationary coefficients")
            # Generate random parameters
            arparams = []
            maparams = []
            ar_order, ma_order = random_order
            for i in range(ar_order):
                arparams.append(2 * np.random.random() - 1)
            for i in range(ma_order):
                maparams.append(2 * np.random.random() - 1)

            # print(arparams)
            arparams = np.array(arparams)
            maparams = np.array(maparams)
            if limit_abs_sum:
                ar_abssum = sum(np.abs(arparams))
                ma_abssum = sum(np.abs(maparams))
                if ar_abssum > 1:
                    arparams = arparams / (ar_abssum + 10e-6)
                    arparams = arparams * (0.5 + 0.5*np.random.random())
                if ma_abssum > 1:
                    maparams = maparams / (ma_abssum + 10e-6)
                    maparams = maparams * (0.5 + 0.5*np.random.random())

            arparams = arparams - np.mean(arparams)
            maparams = maparams - np.mean(maparams)
            arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=samples)
            is_stationary = arma_process.isstationary

    # sample output from ARMA Process
    series = arma_process.generate_sample(samples, scale=noise_std)
    # make zero-mean:
    series = series - np.mean(series)
    return series, arparams, maparams


def init_ar_dataset(n_samples, ar_val, ar_params=None, noise_std=1.0, plot=False, verbose=False, test=None, pad_to=None):
    # AR-Process
    if ar_params is not None:
        ar_val = len(ar_params)
        params = (ar_params, [])
    else:
        params = None

    if pad_to is None:
        inp_size = ar_val
    else:
        inp_size = pad_to

    series, ar, ma = generate_armaprocess_data(
        samples=n_samples+inp_size,
        noise_std=noise_std,
        random_order=(ar_val, 0),
        params=params,
    )
    # print("series mean", np.mean(series))

    if pad_to is not None:
        ar_pad = [0.0] * max(0, pad_to - ar_val)
        ar = list(ar) + ar_pad

    if verbose:
        print("AR params: ")
        print(ar)

    # Initialize data for DAR
    dataset_train, dataset_test, series_train, series_test = create_dataset(
        series=series,
        n_samples=n_samples,
        sample_inp_size=inp_size,
        sample_out_size=1,
        verbose=verbose,
        plot=plot,
        test=test,
    )

    return dataset_train, dataset_test, series_train, series_test, ar


class MaestroDataset(Dataset):
    def __init__(self, midi_files, sequence_length, verbose=True):
        self.sequence_length = sequence_length
        self.data = []
        
        if verbose:
            print(f"\nInitializing MaestroDataset with {len(midi_files)} files")
        
        # Process each MIDI file
        total_sequences = 0
        for midi_file in midi_files:
            try:
                # Extract features from MIDI
                features = self.extract_midi_features(midi_file, verbose=False)
                if len(features) > 0:
                    # Create sequences
                    sequences = self.create_sequences(features)
                    total_sequences += len(sequences)
                    self.data.extend(sequences)
            except Exception as e:
                if verbose:
                    print(f"Error processing {os.path.basename(midi_file)}: {str(e)}")
                
        if len(self.data) == 0:
            raise ValueError("No valid sequences could be created from the MIDI files")
        
        if verbose:
            print(f"Created {total_sequences} total sequences")
        
        try:
            # Convert sequences to numpy arrays first
            x_data = np.array([seq[0] for seq in self.data])  # Keep all features for input
            y_data = np.array([seq[1][0] for seq in self.data])  # Only pitch for target
            
            # Initialize scalers
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            
            # Reshape for normalization
            x_reshaped = x_data.reshape(-1, x_data.shape[-1])
            x_normalized = self.x_scaler.fit_transform(x_reshaped)
            x_data = x_normalized.reshape(x_data.shape)
            
            # Normalize target values
            y_data = self.y_scaler.fit_transform(y_data.reshape(-1, 1))
            
            # Convert numpy arrays to tensors
            self.x_data = torch.from_numpy(x_data).float()  # shape: (n_sequences, sequence_length, 3)
            self.y_data = torch.from_numpy(y_data).float()  # shape: (n_sequences, 1)
            self.length = len(self.data)
            
            if verbose:
                print(f"Dataset initialization complete")
                print(f"Input shape: {self.x_data.shape}")
                print(f"Target shape: {self.y_data.shape}")
                print("\nNormalization Statistics:")
                print("Input Features:")
                print(f"  Mean: {self.x_scaler.mean_}")
                print(f"  Std: {self.x_scaler.scale_}")
                print("\nTarget (Pitch):")
                print(f"  Mean: {self.y_scaler.mean_}")
                print(f"  Std: {self.y_scaler.scale_}")
        except Exception as e:
            print("Error converting sequences to tensors:", str(e))
            raise
    
    def denormalize_predictions(self, predictions):
        """
        Convert normalized predictions back to original scale
        Args:
            predictions: numpy array of normalized predictions
        Returns:
            numpy array of denormalized predictions
        """
        return self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def denormalize_input(self, x_data):
        """
        Convert normalized input data back to original scale
        Args:
            x_data: numpy array of normalized input data
        Returns:
            numpy array of denormalized input data
        """
        x_reshaped = x_data.reshape(-1, x_data.shape[-1])
        x_denormalized = self.x_scaler.inverse_transform(x_reshaped)
        return x_denormalized.reshape(x_data.shape)
    
    def extract_midi_features(self, midi_file, verbose=False):
        try:
            mid = pretty_midi.PrettyMIDI(midi_file)
            note_events = []
            
            for instrument in mid.instruments:
                if not instrument.is_drum:  # Skip drum tracks
                    for note in instrument.notes:
                        pitch = note.pitch 
                        # Additional features
                        velocity = note.velocity 
                        duration = note.end - note.start  # Duration in seconds
                        start_time = note.start  # Start time in seconds
                        
                        note_events.append({
                            'pitch': pitch,
                            'velocity': velocity,
                            'duration': duration,
                            'start_time': start_time
                        })
            
            # Sort notes by start time to maintain temporal order
            note_events.sort(key=lambda x: x['start_time'])
            
            # Convert to numpy arrays
            if note_events:
                max_duration = max(x['duration'] for x in note_events)
                features = np.array([
                    [x['pitch'],  # Main feature
                     x['velocity'],  # Additional features
                     x['duration']] 
                    for x in note_events
                ])
                return features
            return np.array([])
            
        except Exception as e:
            if verbose:
                print(f"Error in extract_midi_features: {str(e)}")
            return np.array([])
    
    def create_sequences(self, features):
        sequences = []
        if len(features) <= self.sequence_length + 1:
            return sequences
            
        try:
            # Pre-allocate numpy arrays for efficiency
            n_sequences = len(features) - self.sequence_length - 1
            seq_in = np.zeros((n_sequences, self.sequence_length, 3))  # Keep all features for input
            seq_out = np.zeros((n_sequences, 1))  # Only pitch for output, ensure it's a 2D array
            
            # Create sequences
            for i in range(n_sequences):
                seq_in[i] = features[i:i + self.sequence_length]  # Input includes all features
                seq_out[i] = features[i + self.sequence_length, 0]  # Output only includes pitch, ensure it's a 2D array
                
            return list(zip(seq_in, seq_out))
        except Exception as e:
            print(f"Error in create_sequences: {str(e)}")
            import traceback
            traceback.print_exc()
            return sequences
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]




def load_maestro_data(maestro_dir, sequence_length, test_split=0.2, verbose=False, plot=False):
    if verbose:
        print(f"\nLoading MIDI files from {os.path.basename(maestro_dir)}")
    
    # Get MIDI files in the directory and all subdirectories
    midi_files = []
    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {maestro_dir}")
    
    if verbose:
        print(f"Found {len(midi_files)} MIDI files in total")
        print("\nDirectory structure:")
        # Print unique directories containing MIDI files
        unique_dirs = sorted(set(os.path.dirname(f) for f in midi_files))
        for dir_path in unique_dirs:
            dir_name = os.path.basename(dir_path)
            n_files = len([f for f in midi_files if os.path.dirname(f) == dir_path])
            print(f"  {dir_name}: {n_files} files")
    
    try:
        # Create dataset
        if verbose:
            print("\nCreating MaestroDataset...")
        dataset = MaestroDataset(midi_files, sequence_length, verbose=verbose)
        if verbose:
            print(f"Dataset created with {len(dataset)} sequences")
        
        # Split into train and test
        train_size = int((1 - test_split) * len(dataset))
        test_size = len(dataset) - train_size
        
        if verbose:
            print(f"\nSplitting dataset:")
            print(f"Total sequences: {len(dataset)}")
            print(f"Train size: {train_size}")
            print(f"Test size: {test_size}")
        
        dataset_train, dataset_test = torch.utils.data.random_split(
            dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Extract series for compatibility with AR format
        series = dataset.x_data.numpy().reshape(-1)  # Flatten all sequences into one series
        if verbose:
            print(f"\nCreated time series with {len(series)} points")
        
        series_train = series[:int(len(series) * (1 - test_split))]
        series_test = series[int(len(series) * (1 - test_split)):]
        
        if verbose:
            print(f"Final dataset statistics:")
            print(f"Train set size: {len(dataset_train)}")
            print(f"Test set size: {len(dataset_test)}")
            print(f"Training series length: {len(series_train)}")
            print(f"Test series length: {len(series_test)}")
        
        if plot:
            # Plot first 200 notes of the series
            plt.figure(figsize=(10, 4))
            plt.plot(series[:200])
            plt.title("First 200 notes of the MIDI series")
            plt.xlabel("Time step")
            plt.ylabel("Normalized pitch")
            plt.show()
            plt.close()
        
        return dataset_train, dataset_test, series_train, series_test
        
    except Exception as e:
        print(f"Error in load_maestro_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def load_data(data_config_in, verbose=False, plot=False):
    data_config = copy.deepcopy(data_config_in)
    data_type = data_config.pop("type")
    data = {
        "type": data_type
    }
    
    if data_type == 'AR':
        data["train"], data["test"], data["series_train"], data["series_test"], data["ar"] = init_ar_dataset(
            **data_config,
            verbose=verbose,
            plot=plot,
        )
        data["pad_to"] = data_config_in["pad_to"]
    elif data_type == 'Maestro':
        maestro_dir = data_config.pop("maestro_dir")
        sequence_length = data_config.pop("sequence_length")
        test_split = data_config.pop("test", 0.2)
        
        try:
            # Load data in format compatible with AR data
            dataset_train, dataset_test, series_train, series_test = load_maestro_data(
                maestro_dir=maestro_dir,
                sequence_length=sequence_length,
                test_split=test_split,
                verbose=verbose,
                plot=plot
            )
            
            data["train"] = dataset_train
            data["test"] = dataset_test
            data["series_train"] = series_train
            data["series_test"] = series_test
            data["pad_to"] = sequence_length  # Use sequence length as pad_to value
            
        except Exception as e:
            raise RuntimeError(f"Error loading Maestro dataset: {str(e)}")
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")
        
    return data


#--------------------------#
#     Model                #
#--------------------------#

class DAR(nn.Module):
    '''
    A neural network for MIDI sequence prediction
    Modified to predict only pitch while using all input features
    '''

    def __init__(self, ar, num_layers=3, d_hidden=256, dropout_rate=0.2):
        # Perform initialization of the pytorch superclass
        super(DAR, self).__init__()
        
        d_in = ar * 3  # ar timesteps × 3 features (pitch, velocity, duration)
        d_out = 1  # Only predict pitch
        self.ar = ar
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        if d_hidden is None and num_layers > 1:
            d_hidden = d_in
            
        # Create dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if self.num_layers == 1:
            self.layer_1 = nn.Linear(d_in, d_out, bias=True)
            assert self.layer_1.weight is not None, "Layer 1 weights are not initialized"
        else:
            # First layer with batch normalization
            self.layer_1 = nn.Sequential(
                nn.Linear(d_in, d_hidden, bias=True),
                nn.BatchNorm1d(d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            # Middle layers with batch normalization
            self.mid_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden, bias=True),
                    nn.BatchNorm1d(d_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ) for _ in range(self.num_layers - 2)
            ])
            
            # Output layer
            self.layer_out = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        Input shape: (batch_size, ar_timesteps, 3)
        Output shape: (batch_size, 1) - only pitch predictions
        '''
        # Flatten the input
        x = x.view(x.size(0), -1)  # shape: (batch_size, ar_timesteps * 3)
        
        # Forward pass through layers
        x = self.layer_1(x)
        
        if self.num_layers > 1:
            for layer in self.mid_layers:
                x = layer(x)
            x = self.layer_out(x)
            
        return x





#--------------------------#
#     Ultils               #
#--------------------------#


def compute_stats_ar(results, ar_params, verbose=False):
    weights = results["weights"]
    error = results["predicted"] - results["actual"]
    stats = {}

    abs_error = np.abs(weights - ar_params)

    symmetric_abs_coeff = np.abs(weights) + np.abs(ar_params)
    stats["sMAPE (AR-coefficients)"] = 100 * np.mean(abs_error / (10e-9 + symmetric_abs_coeff))

    sTPE = 100 * np.sum(abs_error) / (10e-9 + np.sum(symmetric_abs_coeff))
    stats["sTPE (AR-coefficients)"] = sTPE

    # predictions error
    stats["MSE"] = np.mean(error ** 2)

    if verbose:
        print("MSE: {}".format(stats["MSE"]))
        print("sMAPE (AR-coefficients): {:6.3f}".format(stats["sMAPE (AR-coefficients)"]))
        print("sTPE (AR-coefficients): {:6.3f}".format(stats["sTPE (AR-coefficients)"]))
        # print("Relative error: {:6.3f}".format(stats["TP (AR-coefficients)"]))
        # print("Mean relative error: {:6.3f}".format(mean_rel_error))

        print("AR params: ")
        print(ar_params)

        print("Weights: ")
        print(weights)
    return stats


def plot_loss_curve(losses, test_loss=None, epoch_losses=None, show=False, save=False, savedir=None, filename=None):
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Squared Error Loss")
    ax.set_title("Training and Testing Loss Over Time")
    
    # Convert losses to numpy array for easier manipulation
    losses = np.array(losses)
    
    # Calculate moving average for smoother batch loss curve
    window_size = min(50, len(losses) // 10)  # Use smaller window for shorter training
    if window_size > 1:
        batch_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        x_loss = np.arange(window_size-1, len(losses))
    else:
        batch_losses = losses
        x_loss = np.arange(len(losses))
    
    # Plot batch losses with transparency
    plt.plot(x_loss, batch_losses, 'b', alpha=0.3, label='Batch MSE Loss')
    
    if epoch_losses is not None:
        iter_per_epoch = int(len(losses) / len(epoch_losses))
        epoch_ends = int(iter_per_epoch/2) + iter_per_epoch*np.arange(len(epoch_losses))
        plt.plot(epoch_ends, epoch_losses, 'b', label='Epoch Average MSE Loss')
    
    if test_loss is not None:
        plt.hlines(test_loss, xmin=x_loss[0], xmax=x_loss[-1], 
                  colors='r', label=f'Test MSE Loss: {test_loss:.4f}')

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits from 0 to 500
    plt.ylim(0, 5)
    
    if save and savedir is not None and filename is not None:
        save_path = os.path.join(savedir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



def plot_prediction_sample(predicted_dar, predicted_var, predicted_arima, actual, dataset, num_obs=250, model_name="AR-Net", show=False, save=False, savedir=None, filename=None):
    """
    Plot comparison of predictions from different models
    Args:
        predicted_dar: DAR model predictions (normalized)
        predicted_var: VAR model predictions (denormalized)
        predicted_arima: ARIMA model predictions (denormalized)
        actual: actual values (normalized)
        dataset: MaestroDataset instance for denormalization
        num_obs: number of observations to plot
        model_name: name of the DAR model
        show: whether to display the plot
        save: whether to save the plot
        savedir: directory to save the plot
        filename: name of the file to save
    """
    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Ensure we have valid data
    if len(actual) == 0 or len(predicted_dar) == 0:
        print("Warning: Empty data arrays provided to plot_prediction_sample")
        return
        
    # Denormalize DAR predictions and actual data
    predicted_dar_denorm = dataset.denormalize_predictions(predicted_dar)
    actual_denorm = dataset.denormalize_predictions(actual.reshape(-1, 1))
    
    # Calculate start index to show last num_obs points
    start_idx = max(0, len(actual_denorm) - num_obs)
    
    # Plot last num_obs points of actual pitch data
    plt.plot(range(num_obs), actual_denorm[start_idx:start_idx + num_obs], 
            label='Actual Pitch', linestyle='-', markersize=2)
    
    # Plot predictions (last 200 points)
    prediction_range = range(50, num_obs)
    
    # DAR predictions (now denormalized)
    plt.plot(prediction_range, predicted_dar_denorm[-200:], 
            label=f'{model_name} Predictions', marker='x', linestyle='--', markersize=2)
    
    # VAR predictions (already denormalized)
    if predicted_var is not None:
        plt.plot(prediction_range, predicted_var, 
                label='VAR Predictions', marker='^', linestyle=':', markersize=2)
    
    # ARIMA predictions (already denormalized)
    if predicted_arima is not None:
        plt.plot(prediction_range, predicted_arima, 
                label='ARIMA Predictions', marker='s', linestyle='-.', markersize=2)
    
    plt.title(f'Model Predictions Comparison (Last {num_obs} Time Steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Pitch Value')
    plt.legend()
    
    if save and savedir is not None and filename is not None:
        save_path = os.path.join(savedir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_error_scatter(predicted, actual, model_name="AR-Net", show=False, save=False, savedir=None, filename=None):
    # error = predicted - actual
    fig3 = plt.figure()
    fig3.set_size_inches(6, 6)
    plt.scatter(actual, predicted - actual, marker='o', s=10, alpha=0.3)
    plt.legend(["{}-Error".format(model_name)])
    if save and savedir is not None and filename is not None:
        save_path = os.path.join(savedir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_weights(ar_val, weights, ar, model_name="AR-Net", show=False, save=False, savedir=None, filename=None):

    if ar is None:
        # If no AR parameters are provided, just plot the model weights
        df = pd.DataFrame(
            zip(
                list(range(1, ar_val + 1)),
                [model_name] * ar_val,
                list(weights.flatten())  # Flatten weights for plotting
            ),
            columns=["AR-coefficient (lag number)", "model", "value (weight)"]
        )
    else:
        # If AR parameters are provided, plot both true and predicted
        df = pd.DataFrame(
            zip(
                list(range(1, ar_val + 1)) * 2,
                ["AR-Process (True)"] * ar_val + [model_name] * ar_val,
                list(ar) + list(weights.flatten())  # Flatten weights for plotting
            ),
            columns=["AR-coefficient (lag number)", "model", "value (weight)"]
        )
    
    
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AR-coefficient (lag number)", hue="model", y="value (weight)", data=df)
    plt.title(f"{model_name} Weights")
    plt.xlabel("AR-coefficient (lag number)")
    plt.ylabel("Value (weight)")
    plt.legend(title="model")
    
    if save and savedir is not None and filename is not None:
        save_path = os.path.join(savedir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_metrics_comparison(metrics, show=False, save=False, savedir=None, filename=None):
    """
    Create a comprehensive visualization of model metrics
    Args:
        metrics: dictionary containing metrics for all models
        show: whether to display the plot
        save: whether to save the plot
        savedir: directory to save the plot
        filename: name of the file to save
    """
    # Extract metric names and values for each model
    model_names = ['DAR', 'VAR', 'ARIMA']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    
    # Plot 1: MSE
    ax1 = fig.add_subplot(gs[0, 0])
    mse_values = [metrics[f"{model}_MSE"] for model in model_names]
    ax1.bar(model_names, mse_values)
    ax1.set_ylabel('MSE')
    ax1.set_title('Model MSE Comparison')
    
    # Plot 2: Entropy
    ax2 = fig.add_subplot(gs[0, 1])
    entropy_values = [metrics[f"{model}_Entropy"] for model in model_names]
    ax2.bar(model_names, entropy_values)
    ax2.set_ylabel('Entropy')
    ax2.set_title('Model Entropy Comparison')
    
    # Plot 3: NLL
    ax3 = fig.add_subplot(gs[0, 2])
    nll_values = [metrics[f"{model}_NLL"] for model in model_names]
    ax3.bar(model_names, nll_values)
    ax3.set_ylabel('Negative Log Likelihood')
    ax3.set_title('Model NLL Comparison')
    
    # Adjust layout
    plt.tight_layout()
    
    if save and savedir is not None and filename is not None:
        save_path = os.path.join(savedir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_results(results, model_config, data, predicted_var, predicted_arima, actual_data, metrics, model_name="MODEL", show=False, save=True, savedir=None):
    """
    Plot all visualizations with rolling window predictions and entropy analysis
    Args:
        results: dictionary containing model results
        model_config: dictionary containing model configuration
        data: dictionary containing data information
        predicted_var: VAR model predictions
        predicted_arima: ARIMA model predictions
        actual_data: actual values
        metrics: dictionary containing computed metrics for all models
        model_name: name of the model
        show: whether to display plots
        save: whether to save plots
        savedir: directory to save plots
    """
    if save:
        if savedir is None:
            savedir = os.getcwd()
        plots_dir = os.path.join(savedir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plots_dir = None
    
    # Plot loss curve
    if "losses" in results:
        plot_loss_curve(
            losses=results["losses"],
            test_loss=results.get("test_mse", None),
            epoch_losses=results.get("epoch_losses", None),
            show=show,
            save=save,
            savedir=plots_dir,
            filename='loss_curve.png'
        )
    
    # Plot prediction sample
    plot_prediction_sample(
        predicted_dar=results["predicted"],
        predicted_var=predicted_var,
        predicted_arima=predicted_arima,
        actual=actual_data,
        dataset=data["test"].dataset,
        num_obs=250,
        model_name=model_name,
        show=show,
        save=save,
        savedir=plots_dir,
        filename='model_predictions_comparison.png'
    )
    
    # Plot error scatter
    plot_error_scatter(
        results["predicted"],
        results["actual"],
        model_name=model_name,
        show=show,
        save=save,
        savedir=plots_dir,
        filename='error_scatter.png'
    )
    
    # Plot weights if available
    if "weights" in results:
        plot_weights(
            ar_val=model_config["ar"],
            weights=results["weights"],
            ar=data.get("ar", None),
            model_name=model_name,
            save=save,
            savedir=plots_dir,
            filename='weights.png'
        )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        metrics=metrics,
        show=show,
        save=save,
        savedir=plots_dir,
        filename='metrics_comparison.png'
    )



def jsonize(results):
    for key, value in results.items():
        if type(value) is list:
            if type(value[0]) is list:
                results[key] = [["{:8.5f}".format(xy) for xy in x] for x in value]
            else:
                results[key] = ["{:8.5f}".format(x) for x in value]
        else:
            results[key] = "{:8.5f}".format(value)
    return results


def list_of_dicts_2_dict_of_lists(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        res[key] = [d[key] for d in sources]
    return res


def list_of_dicts_2_dict_of_means(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        res[key] = np.mean([d[key] for d in sources])
    return res


def list_of_dicts_2_dict_of_means_minmax(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        values = [d[key] for d in sources]
        res[key] = (np.mean(values), min(values), max(values))
    return res


def get_json_filenames(values, subdir=None):
    ar_filename = get_json_filenames_type("AR", values, subdir)
    dar_filename = get_json_filenames_type("DAR", values, subdir)
    return ar_filename, dar_filename


def get_json_filenames_type(model_type, values, subdir=None):
    filename = 'results/{}{}_{}.json'.format(
        subdir + "/" if subdir is not None else "",
        model_type,
        "-".join([str(x) for x in values]))
    
    return filename


def intelligent_regularization(sparsity):
    if sparsity is not None:
        # best:
        # lam = 0.01 * (1.0 / sparsity - 1.0)
        lam = 0.02 * (1.0 / sparsity - 1.0)
        # lam = 0.05 * (1.0 / sparsity - 1.0)

        # alternatives
        # l1 = 0.02 * (np.log(2) / np.log(1 + sparsity) - 1.0)
        # l1 = 0.1 * (1.0 / np.sqrt(sparsity) - 1.0)
    else:
        lam = 0.0
    return lam


def reduce_dimensions_pca(data, n_components=1):
    """
    Reduce dimensionality of MIDI data using PCA
    Args:
        data: numpy array of shape (n_samples, 3) containing pitch, velocity, duration
        n_components: number of components to keep
    Returns:
        Reduced data of shape (n_samples, n_components)
    """
    
    warnings.filterwarnings('ignore', category=FutureWarning, message=".*force_all_finite.*")
    
    # Reshape if needed
    if len(data.shape) > 2:
        data = data.reshape(-1, 3)
    
    # Apply PCA directly since data is already normalized
    pca = PCA(
        n_components=n_components,
        copy=True,
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=None
    )
    data_reduced = pca.fit_transform(data)
    
    # Print detailed PCA information
    print("\nPCA Analysis Results:")
    print("--------------------")
    print(f"Number of components: {n_components}")
    print(f"Original data shape: {data.shape}")
    print(f"Reduced data shape: {data_reduced.shape}")
    print("\nVariance explained by each component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    # Print feature importance
    print("\nFeature importance in first component:")
    features = ['Pitch', 'Velocity', 'Duration']
    for feature, weight in zip(features, pca.components_[0]):
        print(f"{feature}: {abs(weight):.4f}")
    
    return data_reduced, pca, None  

def fit_auto_arima(data, seasonal=True):
    """
    Fit auto_arima model with enhanced parameters for MIDI data
    """
    
    # Suppress all warnings including ARIMA-specific ones
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings('ignore', 'Covariance matrix calculated using the outer product of gradients')
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        

    model = auto_arima(data,
                      start_p=0, start_q=0, max_p=5, max_q=5,
                      m=12 if seasonal else 1,
                      seasonal=seasonal,
                      d=None,
                      D=None if seasonal else 0,
                             trace=False,  # Set to False to reduce output
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True,
                      random_state=42)
    return model


def predict_arima(model, n_steps):
    """
    Generate predictions using fitted ARIMA model
    """
    return model.predict(n_periods=n_steps)

#--------------------------#
#     Training             #
#--------------------------#



class MIDILoss(nn.Module):
    """
    Modified loss function that only calculates loss for pitch prediction
    while using all three input features (pitch, velocity, duration)
    """
    def __init__(self):
        super(MIDILoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean') 
    
    def forward(self, y_pred, y_true):
        # Only take pitch predictions and targets
        # y_pred and y_true shape: (batch_size, 1)
        pitch_loss = self.mse(y_pred, y_true)
        
        return pitch_loss, {
            'pitch_loss': pitch_loss.item()
        }


def train_batch(model, x, y, optimizer, loss_fn, lambda_value=None, device=None):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss
    if isinstance(loss_fn, MIDILoss):
        loss, feature_losses = loss_fn(y_predict, y)
    else:
        loss = loss_fn(y_predict, y)
        feature_losses = None

    # Regularize
    if lambda_value is not None:
        reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=True, device=device)
        if model.num_layers == 1:
            abs_weights = torch.abs(model.layer_1.weight)
            reg = torch.div(2.0, 1.0 + torch.exp(-3.0*abs_weights.pow(1.0/3.0))) - 1.0
            reg_loss = reg_loss + torch.mean(reg)
        else:
            for layer in model.mid_layers:
                if isinstance(layer, nn.Linear):
                    abs_weights = torch.abs(layer.weight)
                    reg = torch.div(2.0, 1.0 + torch.exp(-3.0*abs_weights.pow(1.0/3.0))) - 1.0
                    reg_loss = reg_loss + torch.mean(reg)

        loss = loss + lambda_value * reg_loss

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if feature_losses is not None:
        return loss.data.item(), feature_losses
    return loss.data.item()


def train(model, loader, loss_fn, lr, epochs, lr_decay, est_sparsity, lambda_delay=None, verbose=False, device=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    losses = []
    feature_losses_history = {
        'pitch': []
    }
    batch_index = 0
    epoch_losses = []
    avg_losses = []
    lambda_value = intelligent_regularization(est_sparsity)
    epoch_predictions = []

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc='Training epochs', leave=True)
    
    for e in epoch_pbar:
        epoch_feature_losses = {
            'pitch': []
        }
        
        # For storing predictions of this epoch
        epoch_pred_list = []
        
        # slowly increase regularization until lambda_delay epoch
        if lambda_delay is not None and e < lambda_delay:
            l_factor = e / (1.0 * lambda_delay)
        else:
            l_factor = 1.0

        # Create progress bar for batches within each epoch
        batch_pbar = tqdm(loader, desc=f'Epoch {e+1}/{epochs}', leave=False)
        for x, y in batch_pbar:
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            
            # Train batch and get losses
            if isinstance(loss_fn, MIDILoss):
                loss, feature_losses = train_batch(
                    model=model, x=x, y=y, optimizer=optimizer,
                    loss_fn=loss_fn, lambda_value=l_factor*lambda_value,
                    device=device
                )
                # Store feature-specific losses
                epoch_feature_losses['pitch'].append(feature_losses['pitch_loss'])
            else:
                loss = train_batch(
                    model=model, x=x, y=y, optimizer=optimizer,
                    loss_fn=loss_fn, lambda_value=l_factor*lambda_value,
                    device=device
                )
            
            epoch_losses.append(loss)
            batch_index += 1
            
            # Store predictions
            with torch.no_grad():
                pred = model(x)
                epoch_pred_list.append(pred.cpu().numpy())  
            
            # Update progress bar
            if isinstance(loss_fn, MIDILoss):
                batch_pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'pitch': f'{feature_losses["pitch_loss"]:.4f}'
                })
            else:
                batch_pbar.set_postfix({'loss': f'{loss:.4f}'})
            
        scheduler.step()
        losses.extend(epoch_losses)
        avg_loss = np.mean(epoch_losses)
        avg_losses.append(avg_loss)
        
        # Store feature-specific losses for this epoch
        if epoch_feature_losses['pitch']:
            feature_losses_history['pitch'].append(np.mean(epoch_feature_losses['pitch']))
        
        # Store predictions for this epoch
        epoch_predictions.append(np.concatenate(epoch_pred_list, axis=0))
        
        epoch_losses = []
        if isinstance(loss_fn, MIDILoss):
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'pitch': f'{feature_losses_history["pitch"][-1]:.4f}'
            })
        else:
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        if verbose:
            print(f"Epoch {e + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")
            if isinstance(loss_fn, MIDILoss):
                print(f"  Pitch Loss: {feature_losses_history['pitch'][-1]:.4f}")
            
    if verbose:
        print(f"Training completed. Total batches: {batch_index}")

    return losses, avg_losses, feature_losses_history, epoch_predictions


def test_batch(model, x, y, loss_fn, device=None):
    # Run forward calculation
    y_predict = model.forward(x)
    
    # Compute loss
    if isinstance(loss_fn, MIDILoss):
        loss, feature_losses = loss_fn(y_predict, y)
    else:
        loss = loss_fn(y_predict, y)
        feature_losses = None

    if feature_losses is not None:
        return y_predict, loss, feature_losses
    return y_predict, loss


def test(model, loader, loss_fn, device=None):
    losses = []
    feature_losses_list = []
    y_vectors = []
    y_predict_vectors = []

    # Add progress bar for testing
    test_pbar = tqdm(loader, desc='Testing', leave=True)
    for x, y in test_pbar:
        # Move data to device
        x = x.to(device)
        y = y.to(device)
        
        if isinstance(loss_fn, MIDILoss):
            y_predict, loss, feature_losses = test_batch(model=model, x=x, y=y, loss_fn=loss_fn, device=device)
            feature_losses_list.append(feature_losses)
        else:
            y_predict, loss = test_batch(model=model, x=x, y=y, loss_fn=loss_fn, device=device)

        # Move tensors to CPU and detach before converting to numpy
        losses.append(loss.detach().cpu().numpy()) 
        y_vectors.append(y.detach().cpu().numpy())  
        y_predict_vectors.append(y_predict.detach().cpu().numpy())  
        
        if isinstance(loss_fn, MIDILoss):
            test_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pitch': f'{feature_losses["pitch_loss"]:.4f}'
            })
        else:
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Clear GPU memory after each batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    losses = np.array(losses)
    y_predict_vector = np.concatenate(y_predict_vectors)
    y_vector = np.concatenate(y_vectors)
    
    # Calculate MSE for each feature if using MIDILoss
    if feature_losses_list:
        feature_mse = {
            'pitch_mse': np.mean([f['pitch_loss'] for f in feature_losses_list])
        }
    else:
        feature_mse = None
    
    mse = np.mean((y_predict_vector - y_vector) ** 2)

    # Final cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return y_predict_vector, losses, mse, feature_mse


def run_train_test(dataset_train, dataset_test, model_config, train_config, verbose=False):
    # Create data loaders with device-specific settings
    data_loader_train = DataLoader(
        dataset=dataset_train, 
        batch_size=train_config["batch"], 
        shuffle=True,
        num_workers=train_config.get("num_workers", 2),
        pin_memory=train_config.get("pin_memory", True)
    )
    data_loader_test = DataLoader(
        dataset=dataset_test, 
        batch_size=len(dataset_test), 
        shuffle=False,
        num_workers=train_config.get("num_workers", 2),
        pin_memory=train_config.get("pin_memory", True)
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    if model_config["ma"] > 0:
        raise NotImplementedError("DARMA not implemented")
    else:
        del model_config["ma"]
        model = DAR(**model_config)
        model = model.to(device)  

    # Use custom MIDI loss function
    loss_fn = MIDILoss()

    # Train and get the resulting loss per iteration
    del train_config["batch"]
    losses, avg_losses, feature_losses_history, epoch_predictions = train(
        model=model,
        loader=data_loader_train,
        loss_fn=loss_fn,
        device=device,  
        **train_config,
        verbose=verbose,
    )

    # Test and get the resulting predicted y values
    y_predict, test_losses, test_mse, feature_mse = test(
        model=model, 
        loader=data_loader_test, 
        loss_fn=loss_fn,
        device=device  
    )

    # Get actual values from test dataset
    all_test_data = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=len(dataset_test), 
        shuffle=False,
        num_workers=train_config.get("num_workers", 2),
        pin_memory=train_config.get("pin_memory", True)
    )
    x_test, y_test = next(iter(all_test_data))
    actual = y_test.cpu().numpy()  # This one needs .cpu() since it's a tensor
    predicted = y_predict  # Already a numpy array from test function

    # Get model weights
    if model.num_layers == 1:
        weights = model.layer_1.weight.detach().cpu().numpy()  # This one needs .cpu() since it's a tensor
    else:
        weights = model.layer_out.weight.detach().cpu().numpy()  # This one needs .cpu() since it's a tensor

    return predicted, actual, np.array(losses), weights, test_mse, avg_losses, feature_losses_history, feature_mse, epoch_predictions, model


def run(data, model_config, train_config, verbose=False):
    if verbose:
        print("################ Model: AR-Net ################")
    start = time.time()
    
    predicted, actual, losses, weights, test_mse, epoch_losses, feature_losses_history, feature_mse, epoch_predictions, model = run_train_test(
        dataset_train=data["train"],
        dataset_test=data["test"],
        model_config=model_config,
        train_config=train_config,
        verbose=verbose,
    )
    
    end = time.time()
    duration = end - start

    if verbose:
        print("Time: {:8.4f}".format(duration))
        print("Final train epoch loss: {:10.2f}".format(epoch_losses[-1]))
        print("Test MSE: {:10.2f}".format(test_mse))
        if feature_mse:
            print("Feature-specific Test MSE:")
            print("  Pitch MSE: {:10.4f}".format(feature_mse['pitch_mse']))

    results = {
        "weights": weights,
        "predicted": predicted,
        "actual": actual,
        "test_mse": test_mse,
        "losses": losses,
        "epoch_losses": epoch_losses,
        "feature_losses": feature_losses_history,
        "feature_mse": feature_mse,
        "epoch_predictions": epoch_predictions,
        "model": model  # Add the trained model to results
    }
    
    stats = {
        "Time (s)": duration,
        "Final Train Loss": epoch_losses[-1],
        "Test MSE": test_mse
    }
    
    if feature_mse:
        stats.update({
            "Pitch MSE": feature_mse['pitch_mse']
        })
    
    # Add AR-specific stats if it's AR data type
    if data["type"] == 'AR':
        ar_stats = compute_stats_ar(results, ar_params=data["ar"], verbose=verbose)
        stats.update(ar_stats)
    elif data["type"] == 'Maestro':
        stats["Data Type"] = "Maestro"
        if "ar" in data:
            stats["Estimated AR Parameters"] = data["ar"]

    return results, stats



#--------------------------#
#     Main                 #
#--------------------------#

def load_config(verbose=False, random=True):
    # load specified settings

    #### Data settings ####
    data_config = {
        "type": 'Maestro',  
        "maestro_dir": '/dss/dsshome1/0F/ra65cat2/adl/maestro-v3.0.0/2004', 
        "sequence_length": 10,  # Set the sequence length
        "ar": None,
        "test": 0.2,  # Set the test split ratio
    }

    #### Model settings ####
    model_config = {
        "ar": data_config["sequence_length"],  # Use sequence length as ar value
        "ma": 0,  
        "num_layers": 3,  
        "d_hidden": 256,  
        "dropout_rate": 0.3 
    }

    #### Train settings ####
    train_config = {
        "lr": 1e-4, 
        "lr_decay": 0.95,  
        "epochs":50,  
        "batch": 64,  
        "est_sparsity": 1,  # 0 = fully sparse, 1 = not sparse
        "lambda_delay": 10,  
    }

    if verbose:
        print("data_config\n", data_config)
        print("model_config\n", model_config)
        print("train_config\n", train_config)

    return data_config, model_config, train_config


def fit_var_model(data, maxlags=5, ic='aic'):
    """
    Fit VAR model to the data
    Args:
        data: numpy array of shape (n_samples, 3) containing pitch, velocity, duration
        maxlags: maximum number of lags to consider
        ic: information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
    Returns:
        Fitted VAR model and the original data
    """
    try:
        # Create VAR model
        model = VAR(data)
        
        # Select optimal lag order
        lag_order = model.select_order(maxlags=5)
        selected_lag = lag_order.selected_orders['aic']
        
        print(f"\nVAR Model Analysis:")
        print("-------------------")
        print(f"Selected lag order: {selected_lag}")
        
        # Fit the model with selected lag order
        results = model.fit(maxlags=selected_lag, ic='aic')
        
        # Print model summary
        print("\nVAR Model Summary:")
        print(results.summary())
        
        return results, data
        
    except Exception as e:
        print(f"Error in fit_var_model: {str(e)}")
        raise

def predict_var(model, data, steps=200):
    """
    Generate predictions using fitted VAR model
    Args:
        model: Fitted VAR model
        data: Original data used for fitting
        steps: Number of steps to forecast
    Returns:
        Predictions of shape (steps, 3) for pitch, velocity, duration
    """
    try:
        # Get the last k_ar observations for forecasting
        last_obs = data[-model.k_ar:]
        
        # Generate forecasts
        forecast = model.forecast(last_obs, steps=steps)
        return forecast
        
    except Exception as e:
        print(f"Error in predict_var: {str(e)}")
        raise

def compute_metrics(predictions, actual, model_name="Model"):
    """
    Compute various evaluation metrics for model predictions
    Args:
        predictions: numpy array of predictions
        actual: numpy array of actual values
        model_name: name of the model for reporting
    Returns:
        Dictionary containing all computed metrics
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    actual = np.array(actual)
    
    # Basic error metrics
    mse = np.mean((predictions - actual) ** 2)
    
    # Compute entropy of predictions
    # First, create histogram of predictions
    hist, bins = np.histogram(predictions, bins=50, density=True)
    # Normalize histogram to create probability distribution
    hist = hist / np.sum(hist)
    # Remove zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    # Compute Negative Log Likelihood (NLL)
    # Assuming Gaussian distribution for errors
    sigma = np.std(predictions - actual)  # Standard deviation of errors
    nll = -np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((predictions - actual)**2) / (sigma**2))
    
    metrics = {
        f"{model_name}_MSE": mse,
        f"{model_name}_Entropy": entropy,
        f"{model_name}_NLL": nll
    }
    
    return metrics

def evaluate_models(predictions_dar, predictions_var, predictions_arima, actual_data, verbose=True):
    """
    Evaluate all three models using various metrics
    Args:
        predictions_dar: DAR model predictions
        predictions_var: VAR model predictions
        predictions_arima: ARIMA model predictions
        actual_data: actual values
        verbose: whether to print results
    Returns:
        Dictionary containing all metrics for all models
    """
    # Get the last 200 points of actual data and DAR predictions for fair comparison
    actual_data_last_200 = actual_data[-200:]
    predictions_dar_last_200 = predictions_dar[-200:]
    
    # Compute metrics for each model using last 200 points
    dar_metrics = compute_metrics(predictions_dar_last_200, actual_data_last_200, "DAR")
    var_metrics = compute_metrics(predictions_var, actual_data_last_200, "VAR")
    arima_metrics = compute_metrics(predictions_arima, actual_data_last_200, "ARIMA")
    
    # Combine all metrics
    all_metrics = {**dar_metrics, **var_metrics, **arima_metrics}
    
    if verbose:
        print("\nModel Evaluation Metrics:")
        print("=========================")
        print(f"DAR predictions shape: {predictions_dar.shape}")
        print(f"DAR predictions last 200 shape: {predictions_dar_last_200.shape}")
        print(f"VAR predictions shape: {predictions_var.shape}")
        print(f"ARIMA predictions shape: {predictions_arima.shape}")
        print(f"Actual data shape: {actual_data.shape}")
        print(f"Actual data last 200 shape: {actual_data_last_200.shape}")
        print("\nMetrics (computed on last 200 points for all models):")
        for metric_name, value in all_metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    return all_metrics

def convert_numpy_types(obj):
    """
    Convert NumPy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main(verbose=False, plot=False, save=False, random_ar_param=True):
    # Suppress all warnings including ResourceWarning
    warnings.simplefilter("ignore", ResourceWarning)
    warnings.filterwarnings("ignore")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    @contextlib.contextmanager
    def temp_dir_manager():
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Use the context manager for temporary directory
    with temp_dir_manager():
        try:
            # load configuration dicts
            data_config, model_config, train_config = load_config(verbose, random_ar_param)
            
            # loads Maestro dataset
            data = load_data(data_config, verbose, plot)
            
            # Get the test data in its original 3D form
            test_loader = DataLoader(dataset=data["test"], batch_size=len(data["test"]), shuffle=False)
            x_test, y_test = next(iter(test_loader))
            test_data = x_test.numpy()
            actual_data = y_test.numpy().flatten()
            
            print("\nAnalyzing data structure with PCA:")
            print("================================")
            
            # First analyze with all components to see full variance distribution
            print("\nAnalyzing with all components:")
            _, pca_full, _ = reduce_dimensions_pca(test_data, n_components=3)
            
            # Then proceed with single component for ARIMA
            print("\nReducing to single component for ARIMA:")
            data_arima, pca, _ = reduce_dimensions_pca(test_data, n_components=1)
            
            # Reshape test data for VAR (n_samples, 3)
            var_data = test_data.reshape(-1, 3)
            
            # Split data for training and prediction
            train_size = len(var_data) - 200
            var_train = var_data[:train_size]
            
            # Fit VAR model
            var_model, var_train_data = fit_var_model(var_train, maxlags=15)
            
            # Generate VAR predictions
            var_predictions = predict_var(var_model, var_train_data, steps=200)
            var_predictions_pitch = var_predictions[:, 0]
            
            # Fit ARIMA model
            arima_model = fit_auto_arima(data_arima[:train_size].flatten(), seasonal=False)
            arima_predictions = predict_arima(arima_model, 200)
            
            # Transform ARIMA predictions back to original dimension
            arima_predictions_original = pca.inverse_transform(arima_predictions.reshape(-1, 1))
            arima_predictions_pitch = arima_predictions_original[:, 0]
            
            if verbose:
                print(f"\nModel Details:")
                print(f"-------------------")
                print(f"Training data shape: {var_train.shape}")
                print(f"VAR Predictions shape: {var_predictions.shape}")
                print(f"ARIMA Predictions shape: {arima_predictions_pitch.shape}")
                print(f"Number of lags used in VAR: {var_model.k_ar}")
                print("\nARIMA Model Summary:")
                print(arima_model.summary())
                
                # runs training and testing for DAR model
                results_dar, stats_dar = run(data, model_config, train_config, verbose)
                
                # Denormalize predictions from all models
                dar_predictions = data["test"].dataset.denormalize_predictions(results_dar["predicted"])
                var_predictions_pitch = data["test"].dataset.denormalize_predictions(var_predictions_pitch.reshape(-1, 1))
                arima_predictions_pitch = data["test"].dataset.denormalize_predictions(arima_predictions_pitch.reshape(-1, 1))
                
                # Evaluate models
                metrics = evaluate_models(
                    predictions_dar=dar_predictions,
                    predictions_var=var_predictions_pitch,
                    predictions_arima=arima_predictions_pitch,
                    actual_data=actual_data,
                    verbose=True
                )
                
                # optional plotting
                if plot:
                    os.makedirs('plots', exist_ok=True)
                    
                    # Plot all results including metrics
                    plot_results(
                        results=results_dar,
                        model_config=model_config,
                        data=data,
                        predicted_var=var_predictions_pitch,
                        predicted_arima=arima_predictions_pitch,
                        actual_data=actual_data,
                        metrics=metrics,  # Pass the already computed metrics
                        model_name="AR-Net",
                        show=False,
                        save=True,
                        savedir='plots'
                    )

                # Save metrics to a file
                if save:
                    metrics_file = os.path.join('plots', 'model_metrics.json')
                    # Convert NumPy types to native Python types before saving
                    metrics_converted = convert_numpy_types(metrics)
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics_converted, f, indent=4)
                    print(f"\nMetrics saved to {metrics_file}")
                    
        except RuntimeError as e:
            print(f"Runtime error: {str(e)}")
            if device.type == "cuda":
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                torch.cuda.empty_cache()
            raise
        finally:
            # Clean up GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
                if verbose:
                    print("\nCleaned up GPU memory")
                    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                    print(f"Final GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(verbose=True, plot=True, save=True, random_ar_param=False)