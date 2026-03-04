import dpp
import torch
import torch.nn as nn

from torch.distributions import Categorical

from dpp.data.batch import Batch
from dpp.utils import diff

class Hamming_Embedding(nn.Module):
    def __init__(self, input_length, d_model):
        super().__init__()
        self.input_length = input_length
        self.d_model = d_model
        # Register reference strings as a buffer (non-trainable)
        self.register_buffer('references', torch.randint(0, 2, (d_model, input_length), dtype=torch.int8))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_length, f"Input shape mismatch: expected {self.input_length}, got {x.shape[-1]}"
        assert x.dtype == torch.int8, f"Input type mismatch: expected torch.int8, got {x.dtype}"
        # x shape: (batch_size, input_length)
        # Expand dimensions to compute Hamming distance via broadcasting
        hamming_dist = (x.unsqueeze(1) != self.references.unsqueeze(0)).float().sum(dim=-1)

        # if self.training:
        #     # Add Gaussian noise to the Hamming distance
        #     # The noise is generated with mean=0 and 3*sigma=0.25
        #     sigma = 0.25 / 3
        #     noise = torch.randn_like(hamming_dist) * sigma  # Generate Gaussian noise
        #     hamming_dist += noise  # Add noise to the distances
        
        mean = self.input_length / 2
        std = 0.5 * (self.input_length ** 0.5)  # Standard deviation for Hamming distance
        return (hamming_dist - mean) / std  # Normalize by subtracting the mean
    

class RNNLastOutput(nn.Module):
    """
    A wrapper for an RNN module that returns only the last output time step
    and includes a dedicated weight initialization method.
    """
    def __init__(self, rnn_module):
        super().__init__()
        self.rnn = rnn_module

    def forward(self, x):
        output_sequence, _ = self.rnn(x)
        last_output = output_sequence[:, -1, :]
        return last_output

    def init_rnn_weights(self, gain=1.0):
        """
        Initializes the weights of the wrapped RNN module.
        
        Args:
            gain (float): The gain value for the initialization functions.
        """
        # Iterate over the named parameters of the wrapped RNN module
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0)
            elif 'weight_hh' in name:
                # Use orthogonal initialization for hidden-to-hidden weights (recurrent connections)
                nn.init.orthogonal_(param, gain=int(gain))
            elif 'weight_ih' in name:
                # Use Xavier uniform initialization for input-to-hidden weights
                nn.init.xavier_uniform_(param, gain=gain)
    
class ParallelRNNExtractor(nn.Module):
    """
    Applies a separate RNN to each process/dimension of the input
    and concatenates their final hidden states.
    """
    def __init__(self, num_processes, context_size, rnn_type="gru"):
        super().__init__()
        # Ensure context_size is divisible by num_processes
        if context_size % num_processes != 0:
            raise ValueError(
                f"context_size ({context_size}) must be divisible by "
                f"num_processes ({num_processes}) for ParallelRNNExtractor."
            )
        
        self.num_processes = num_processes
        self.context_size = context_size
        self.rnn_hidden_size_per_process = context_size // num_processes

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        
        self.rnn_list = nn.ModuleList()
        for _ in range(self.num_processes):
            rnn_module = rnn_cls(
                input_size=1,  # Each RNN processes 1 dimension
                hidden_size=self.rnn_hidden_size_per_process,
                num_layers=1,
                batch_first=True,
            )
            rnn_wrapper = RNNLastOutput(rnn_module)
            # Apply weight initialization
            rnn_wrapper.init_rnn_weights(gain=1.0)
            self.rnn_list.append(rnn_wrapper)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, history_length, num_processes)
        
        Returns:
            context: Output tensor of shape (batch_size, context_size)
        """
        # Split the input tensor along the last dimension (num_processes)
        # This creates a tuple of `num_processes` tensors,
        # each with shape (batch_size, history_length, 1)
        inputs = x.split(1, dim=-1)
        
        outputs = []
        for i in range(self.num_processes):
            # inputs[i] shape: (batch_size, history_length, 1)
            # rnn_output shape: (batch_size, rnn_hidden_size_per_process)
            rnn_output = self.rnn_list[i](inputs[i])
            outputs.append(rnn_output)
            
        # Concatenate all outputs along the feature dimension
        # Resulting shape: (batch_size, num_processes * rnn_hidden_size_per_process)
        # which is equal to (batch_size, context_size)
        context = torch.cat(outputs, dim=1)
        return context
    

class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 64, # Size of the context embedding (history embedding)
        history_length: int = 16, # Length of the history to use for the model
        num_processes: int = 1, # Number of parallel processes to use for training
        mean_log_inter_time_source: float = 0.0,
        std_log_inter_time_source: float = 1.0,
        context_extractor: str = "mlp", # How to extract features from history, possible choices {"mlp", "gru", "lstm"}
    ):
        super().__init__()
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.num_processes = num_processes
        self.mean_log_inter_time_source = mean_log_inter_time_source
        self.std_log_inter_time_source = std_log_inter_time_source
        
        if context_extractor not in {"mlp", "gru", "lstm"}:
            raise ValueError(f"Unknown context_extractor: {context_extractor}")
        if context_extractor == "mlp":
            self.context_constructor = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(history_length*num_processes, self.context_size),
            )
            # Initialization function
            def init_mlp_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            # Apply initialization
            self.context_constructor.apply(init_mlp_weights)
        elif context_extractor in {"gru", "lstm"}:
            if self.context_size % self.num_processes != 0:
                raise ValueError(
                    f"For 'gru' or 'lstm' extractor, context_size ({self.context_size}) "
                    f"must be divisible by num_processes ({self.num_processes})."
                )
            self.context_constructor = ParallelRNNExtractor(
                num_processes=self.num_processes,
                context_size=self.context_size,
                rnn_type=context_extractor
            )

         # Define means and stds as tensors
        means = torch.tensor([mean_log_inter_time, mean_log_inter_time_source])
        stds = torch.tensor([std_log_inter_time, std_log_inter_time_source])

        # Register them as non-trainable buffers
        # This ensures they are part of the model's state_dict and moved to the correct device
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def get_mean_log_inter_time(self):
        """
        Returns the mean of log-inter-event-times.
        """
        return self.mean_log_inter_time

    def get_std_log_inter_time(self):
        """
        Returns the std of log-inter-event-times.
        """
        return self.std_log_inter_time
    
    def get_context(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, history_length, number of neurons)

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, context_size) 

        """
        assert features.dim() == 3, f"Expected features to have shape (batch_size, history_length, number of neurons), got {features.shape}"
        # Standardize using broadcasting.
        # self.means has shape (2,), features has shape (B, H, 2)
        # Broadcasting aligns the last dimension, performing the operation element-wise.
        # This operation creates a NEW tensor, so the original `features` is not modified.
        if self.num_processes == 1:
            context = (features - self.means[0]) / self.stds[0]
        else:
            context = (features - self.means) / self.stds

        # context = context.view(context.shape[0], -1)  # (batch_size, history_length * number of neurons)
        context = self.context_constructor(context)  # Apply linear transformation and non-linearity
        return context
       

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size)

        """
        raise NotImplementedError()
    

    def log_prob_next(self, history: torch.Tensor, target:  torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood for the next inter-event time.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        context = self.get_context(history) # it add an init context to the front
        
        inter_time_dist = self.get_inter_time_dist(context)
        log_p = inter_time_dist.log_prob(target)  # (batch_size, seq_len)
        #log_surv = inter_time_dist.log_survival_function(target) # should I add it or not?
        # print("log_p.mean():", log_p.mean(), "log_surv.mean():", log_surv.mean())
        return log_p #+ log_surv
    

    def mean_next_inter_time(self, history: torch.Tensor) -> torch.Tensor:
        context = self.get_context(history)
        inter_time_dist = self.get_inter_time_dist(context)
        return inter_time_dist.mean
    
    def sample_next_inter_time_dist(self, history: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
        context = self.get_context(history)
        inter_time_dist = self.get_inter_time_dist(context)
        samples = inter_time_dist.sample(sample_shape=(num_samples,))  # (num_samples, batch_size)
        samples = samples.transpose(0, 1)  # (batch_size, num_samples), so that each row corresponds to a batch element
        return samples  # (batch_size, num_samples)
    
    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = self.context_init
        else:
            # Use the provided context vector
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0)

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(next_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)

            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        return Batch(inter_times=inter_times, mask=mask, marks=None)
