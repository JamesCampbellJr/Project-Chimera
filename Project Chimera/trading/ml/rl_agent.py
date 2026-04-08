"""
Reinforcement Learning agent for Solana trading (PyTorch).

Implements a Dueling DQN with experience replay, target-network soft
updates, and epsilon-greedy exploration with exponential decay.

Actions
-------
0 = HOLD, 1 = BUY, 2 = SELL

The optional continuous position-sizing component maps the Q-value
magnitude to a fraction of available capital.
"""

from __future__ import annotations

import logging
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


NUM_ACTIONS = len(Action)

# ---------------------------------------------------------------------------
# Experience tuple
# ---------------------------------------------------------------------------

class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network.

    Splits into a *value* stream and an *advantage* stream, recombined as::

        Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        # Shared feature extractor
        layers: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)            # (B, 1)
        advantage = self.advantage_stream(features)    # (B, A)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def build_state(
    price_features: np.ndarray,
    position_info: np.ndarray,
    account_state: np.ndarray,
) -> np.ndarray:
    """Concatenate sub-vectors into a single flat state vector.

    Parameters
    ----------
    price_features : 1-D array of market / technical features.
    position_info  : [position_size, entry_price, unrealised_pnl, holding_time].
    account_state  : [balance, equity, margin_used].
    """
    return np.concatenate([
        price_features.ravel(),
        position_info.ravel(),
        account_state.ravel(),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Trading RL Agent
# ---------------------------------------------------------------------------

@dataclass
class TradingRLAgent:
    """DQN trading agent with target network and epsilon-greedy policy.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the flattened state vector.
    hidden_dim : int
        Hidden layer width of the Dueling DQN.
    num_layers : int
        Number of hidden layers in the shared feature extractor.
    gamma : float
        Discount factor.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size for replay sampling.
    buffer_capacity : int
        Maximum replay buffer size.
    epsilon_start : float
        Initial exploration probability.
    epsilon_end : float
        Minimum exploration probability.
    epsilon_decay : float
        Exponential decay rate per step.
    tau : float
        Soft-update coefficient for target network.
    position_sizing : bool
        If ``True``, map normalised Q-value magnitude to a continuous
        position size in ``(0, 1]``.
    device : str
        PyTorch device string.
    """

    state_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    tau: float = 0.005
    position_sizing: bool = False
    device: str = "cpu"

    # Internal state
    _policy_net: DuelingDQN = field(init=False, repr=False)
    _target_net: DuelingDQN = field(init=False, repr=False)
    _optimizer: Adam = field(init=False, repr=False)
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _epsilon: float = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _steps: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        self._policy_net = DuelingDQN(
            self.state_dim, NUM_ACTIONS, self.hidden_dim, self.num_layers,
        ).to(self._device)
        self._target_net = DuelingDQN(
            self.state_dim, NUM_ACTIONS, self.hidden_dim, self.num_layers,
        ).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = Adam(self._policy_net.parameters(), lr=self.lr)
        self._buffer = ReplayBuffer(self.buffer_capacity)
        self._epsilon = self.epsilon_start
        logger.info(
            "TradingRLAgent ready: state_dim=%d  params=%d",
            self.state_dim,
            sum(p.numel() for p in self._policy_net.parameters()),
        )

    # ------------------------------------------------------------------ #
    # State construction helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_state(
        market_data: np.ndarray,
        position: Optional[np.ndarray] = None,
        account: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build a state vector from components (convenience wrapper)."""
        pos = position if position is not None else np.zeros(4, dtype=np.float32)
        acc = account if account is not None else np.zeros(3, dtype=np.float32)
        return build_state(market_data, pos, acc)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(
        self, state: np.ndarray, greedy: bool = False,
    ) -> Tuple[int, float]:
        """Epsilon-greedy action selection.

        Returns ``(action_idx, position_size)``.  ``position_size`` is
        ``1.0`` when ``self.position_sizing`` is disabled.
        """
        if not greedy and random.random() < self._epsilon:
            action = random.randrange(NUM_ACTIONS)
            return action, 1.0

        self._policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            q_values = self._policy_net(state_t).squeeze(0)
            action = int(q_values.argmax().item())

        size = 1.0
        if self.position_sizing:
            # Sigmoid of the selected Q-value maps to (0, 1]
            size = float(torch.sigmoid(q_values[action]).item())
            size = max(size, 0.01)  # floor

        return action, size

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self._buffer.push(Experience(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient update.

        Returns the loss value, or *None* if the buffer is too small.
        """
        if len(self._buffer) < self.batch_size:
            return None

        batch = self._buffer.sample(self.batch_size)
        states = torch.tensor(
            np.array([e.state for e in batch]), dtype=torch.float32, device=self._device,
        )
        actions = torch.tensor(
            [e.action for e in batch], dtype=torch.long, device=self._device,
        )
        rewards = torch.tensor(
            [e.reward for e in batch], dtype=torch.float32, device=self._device,
        )
        next_states = torch.tensor(
            np.array([e.next_state for e in batch]), dtype=torch.float32, device=self._device,
        )
        dones = torch.tensor(
            [e.done for e in batch], dtype=torch.float32, device=self._device,
        )

        # Current Q
        self._policy_net.train()
        q_values = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self._policy_net(next_states).argmax(dim=1)
            next_q = self._target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm=1.0)
        self._optimizer.step()

        # Epsilon decay
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
        self._steps += 1

        # Soft update target network
        self._soft_update()

        return float(loss.item())

    def _soft_update(self) -> None:
        for tp, pp in zip(self._target_net.parameters(), self._policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def update_target_network(self) -> None:
        """Hard copy of policy weights to target network."""
        self._target_net.load_state_dict(self._policy_net.state_dict())
        logger.debug("Target network hard-updated at step %d", self._steps)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_policy(self, path: str) -> None:
        """Save policy network, optimizer state, and hyper-params."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state": self._policy_net.state_dict(),
                "target_state": self._target_net.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "epsilon": self._epsilon,
                "steps": self._steps,
                "config": {
                    "state_dim": self.state_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "gamma": self.gamma,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "buffer_capacity": self.buffer_capacity,
                    "epsilon_start": self.epsilon_start,
                    "epsilon_end": self.epsilon_end,
                    "epsilon_decay": self.epsilon_decay,
                    "tau": self.tau,
                    "position_sizing": self.position_sizing,
                },
            },
            path,
        )
        logger.info("Policy saved to %s (step %d, ε=%.4f)", path, self._steps, self._epsilon)

    @classmethod
    def load_policy(cls, path: str, device: str = "cpu") -> "TradingRLAgent":
        """Load a saved agent."""
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        agent = cls(**cfg, device=device)
        agent._policy_net.load_state_dict(data["policy_state"])
        agent._target_net.load_state_dict(data["target_state"])
        agent._optimizer.load_state_dict(data["optimizer_state"])
        agent._epsilon = data["epsilon"]
        agent._steps = data["steps"]
        logger.info("Policy loaded from %s (step %d, ε=%.4f)", path, agent._steps, agent._epsilon)
        return agent
