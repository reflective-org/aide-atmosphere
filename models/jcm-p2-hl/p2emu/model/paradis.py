"""Paradis neural architecture."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock


class VerticalMixing(nn.Module):
    """Lightweight MLP across pressure levels for explicit vertical coupling.

    Applied after the output projection. For each atmospheric variable,
    mixes information across the level dimension via a small residual MLP.
    Surface variables pass through unchanged.
    """

    def __init__(self, num_atm_vars: int, num_levels: int, num_surface_vars: int,
                 hidden_mult: int = 2):
        super().__init__()
        self.num_atm_vars = num_atm_vars
        self.num_levels = num_levels
        self.num_surface_vars = num_surface_vars
        self.num_atm_features = num_atm_vars * num_levels

        self.level_mlp = nn.Sequential(
            nn.Linear(num_levels, num_levels * hidden_mult),
            nn.SiLU(),
            nn.Linear(num_levels * hidden_mult, num_levels),
        )
        # Initialize near-zero so residual starts as identity
        nn.init.zeros_(self.level_mlp[-1].weight)
        nn.init.zeros_(self.level_mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atm = x[:, :self.num_atm_features]
        sfc = x[:, self.num_atm_features:]

        B, _, H, W = atm.shape
        # Reshape to [B, num_vars, num_levels, H, W]
        atm = atm.view(B, self.num_atm_vars, self.num_levels, H, W)
        # MLP across levels: permute levels to last dim, apply, permute back
        atm = atm + self.level_mlp(atm.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        atm = atm.view(B, -1, H, W)

        return torch.cat([atm, sfc], dim=1)


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class Paradis(nn.Module):
    """Paradis model adapted for shallow water equations."""

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        # JCM PORT: this guard was vacuous -- it compared a hardcoded literal to itself, so it
        # could never fire and silently permitted any grid. The real requirement lives in
        # NeuralSemiLagrangian, which now checks the latitude spacing it actually depends on and
        # switches to an exact monotonic mapping when needed. JCM is T63 Gaussian.
        self.grid = cfg.data.get("grid", "gaussian") if hasattr(cfg, "data") else "gaussian"

        mesh_size = (self.nlat, self.nlon)

        hidden_dim = cfg.model.get("latent_size")

        self.num_vels = cfg.model.get("velocity_vectors")
        diffusion_size = cfg.model.get("diffusion_size")
        reaction_size = cfg.model.get("reaction_size")

        adv_interpolation = cfg.model.get("adv_interpolation")
        bias_channels = cfg.model.get("bias_channels", 4)
        num_encoder_layers = cfg.model.get("num_encoder_layers", 1)

        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = get_scaled_timestep(cfg.model.get("base_dt")) / self.num_layers

        # Input projection
        self.activation_function = nn.SiLU

        input_dim = (
            datamodule.dataset.num_in_dyn_features
            + datamodule.dataset.num_in_static_features
        )

        current_dim = input_dim
        encoder_layers = []
        bias = False
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=True)

            # Initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)

            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)

            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())

            current_dim = hidden_dim

        fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)

        self.input_proj = nn.Sequential(*encoder_layers)

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=hidden_dim,
                    kernel_size=3,
                    mesh_size=mesh_size,
                    bias_channels=bias_channels,
                    pre_normalize=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    hidden_dim,
                    mesh_size,
                    num_vels=self.num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                    project_advection=cfg.model.get("projected_advection", True),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=["CLinear"] * 2,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reaction_size,
                    kernel_size=1,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            layers=["SepConv", "CLinear"],
            input_dim=hidden_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=hidden_dim,
            mesh_size=mesh_size,
            kernel_size=3,
            activation=False,
            bias_channels=bias_channels,
        )

        # Gradient checkpointing (set by trainer based on config)
        self.use_gradient_checkpointing = False

        # Vertical mixing (set by trainer if enabled)
        self.vertical_mixing = None

        # Soft output clamping bounds (set by trainer if enabled)
        self.register_buffer("output_clamp_min", None)
        self.register_buffer("output_clamp_max", None)
        self.soft_clamp = False

    def _layer_forward(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Forward pass for a single advection-diffusion-reaction layer."""
        batch_size = hidden.shape[0]

        velocities_raw = self.velocity_nets[layer_idx](hidden)
        velocities = velocities_raw.reshape(
            batch_size, 2, self.num_vels, self.nlat, self.nlon
        )
        u = velocities[:, 0]
        v = velocities[:, 1]

        advected = self.advection[layer_idx](hidden, u, v, self.dt)
        hidden = hidden + advected

        diffused = self.diffusion[layer_idx](hidden)
        hidden = hidden + diffused

        reacted = self.reaction[layer_idx](hidden)
        hidden = hidden + reacted

        return hidden

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fields and winds.

        Parameters
        ----------
        fields : torch.Tensor
            Input fields of shape (batch, in_channels, nlat, nlon)
        Returns
        -------
        torch.Tensor
            Output fields of shape (batch, out_channels, nlat, nlon)
        """
        x = fields

        # Project features to latent space
        hidden = self.input_proj(x)

        for i in range(self.num_layers):
            if self.use_gradient_checkpointing and self.training:
                hidden = grad_checkpoint(
                    self._layer_forward, hidden, i, use_reentrant=False
                )
            else:
                hidden = self._layer_forward(hidden, i)

        # Project back to physical space
        output = self.output_proj(hidden)

        # Vertical mixing across pressure levels
        if self.vertical_mixing is not None:
            output = self.vertical_mixing(output)

        # Soft output clamping
        if self.soft_clamp and self.output_clamp_min is not None:
            cmin = self.output_clamp_min.view(1, -1, 1, 1)
            cmax = self.output_clamp_max.view(1, -1, 1, 1)
            output = cmin + (cmax - cmin) * torch.sigmoid(output)

        return output