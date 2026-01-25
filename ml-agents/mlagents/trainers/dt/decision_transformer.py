"""
Decision Transformer architecture for offline reinforcement learning

Based on "Decision Transformer: Reinforcement Learning via Sequence Modeling"
https://arxiv.org/abs/2106.01345
"""

from typing import Optional
from mlagents.torch_utils import torch, nn, default_device
import torch.nn.functional as F
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class DecisionTransformer(nn.Module):
    """
    Decision Transformer: Treats RL as a sequence modeling problem

    Given a sequence of (return-to-go, state, action), predicts the next action
    Enables offline RL from logged data and conditioning on desired performance
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        dropout: float = 0.1,
        max_timestep: int = 4096,
        action_tanh: bool = True,
    ):
        """
        Initialize Decision Transformer

        :param state_dim: Dimension of state/observation
        :param action_dim: Dimension of action space
        :param hidden_dim: Hidden dimension for transformer
        :param num_layers: Number of transformer layers
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        :param max_timestep: Maximum timestep for positional encoding
        :param action_tanh: Whether to use tanh activation for actions
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_tanh = action_tanh

        # Embeddings for each modality
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_timestep, hidden_dim)

        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() if action_tanh else nn.Identity(),
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(
            f"DecisionTransformer initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}"
        )

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of Decision Transformer

        :param states: (batch, seq_len, state_dim)
        :param actions: (batch, seq_len, action_dim)
        :param returns_to_go: (batch, seq_len, 1)
        :param timesteps: (batch, seq_len) - integer timesteps
        :param attention_mask: Optional attention mask
        :return: Predicted actions (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        state_embeds = self.state_embed(states)  # (batch, seq_len, hidden)
        action_embeds = self.action_embed(actions)  # (batch, seq_len, hidden)
        return_embeds = self.return_embed(returns_to_go)  # (batch, seq_len, hidden)

        # Timestep embedding
        time_embeds = self.timestep_embed(timesteps)  # (batch, seq_len, hidden)

        # Stack embeddings: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        # Shape: (batch, seq_len, 3, hidden)
        stacked = torch.stack([return_embeds, state_embeds, action_embeds], dim=2)

        # Reshape to (batch, 3*seq_len, hidden)
        stacked = stacked.reshape(batch_size, 3 * seq_len, self.hidden_dim)

        # Add positional encoding (repeat for R, s, a)
        time_embeds_repeated = time_embeds.repeat_interleave(3, dim=1)
        stacked = stacked + time_embeds_repeated

        # Layer norm
        stacked = self.embed_ln(stacked)

        # Create attention mask for causal modeling (if not provided)
        if attention_mask is None:
            # Causal mask: can only attend to past and current tokens
            attention_mask = torch.triu(
                torch.ones(3 * seq_len, 3 * seq_len), diagonal=1
            ).bool()
            attention_mask = attention_mask.to(stacked.device)

        # Transformer forward
        transformer_output = self.transformer(stacked, mask=attention_mask)

        # Extract state positions (we predict action from state)
        # Positions: 1, 4, 7, ... (every third position starting from 1)
        state_positions = torch.arange(1, 3 * seq_len, 3, device=stacked.device)
        state_output = transformer_output[:, state_positions, :]

        # Predict actions
        action_preds = self.action_head(state_output)

        return action_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """
        Get action for current state (inference mode)

        :param states: (batch, seq_len, state_dim)
        :param actions: (batch, seq_len, action_dim)
        :param returns_to_go: (batch, seq_len, 1)
        :param timesteps: (batch, seq_len)
        :return: Action for last timestep (batch, action_dim)
        """
        # Forward pass
        action_preds = self.forward(states, actions, returns_to_go, timesteps)

        # Return action for last timestep
        return action_preds[:, -1, :]
