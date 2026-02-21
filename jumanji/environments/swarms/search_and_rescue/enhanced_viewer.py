# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced viewer for SearchAndRescue that visualises the full agent logic chain:
observation → action → result, per agent, per frame.

Layout (GridSpec):
  Row 0 (height 3): Main map — full width
  Row 1 (height 2): Per-agent panels, each agent gets 2 columns:
                    [obs_radar_i | action_bars_i] × num_agents
"""

from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from jumanji.environments.swarms.search_and_rescue.types import Observation, State


# ── colour palette ────────────────────────────────────────────────────────────
_AGENT_COLORS = [
    "#3A7BD5",  # blue  – agent 0
    "#E84545",  # red   – agent 1
    "#2ECC71",  # green – agent 2
    "#F39C12",  # amber – agent 3
    "#9B59B6",  # purple– agent 4
]
_TARGET_UNFOUND = "#E98449"
_TARGET_FOUND = "#B3B6BC"
_CH0_COLOR = "#A8DADC"  # other agents in obs
_CH1_COLOR = "#B3B6BC"  # found targets in obs
_CH2_COLOR = "#FF6B35"  # unfound targets in obs
_ROT_COLOR = "#9B59B6"  # rotation action bar
_ACC_COLOR = "#27AE60"  # acceleration action bar


def _agent_color(i: int) -> str:
    return _AGENT_COLORS[i % len(_AGENT_COLORS)]


# ── helpers ───────────────────────────────────────────────────────────────────


def _lighten(hex_color: str, factor: float = 0.35) -> Tuple[float, float, float, float]:
    """Return an RGBA tuple that is a lighter (more transparent) version of hex_color."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r, g, b, factor)


def _draw_fov_wedge(
    ax: plt.Axes,
    pos: np.ndarray,
    heading: float,
    radius: float,
    view_angle_rad: float,
    color: str,
    alpha: float,
) -> None:
    """Draw a filled FOV wedge (fan) on *ax*.

    Args:
        pos: (x, y) centre of the agent.
        heading: agent heading in radians (0 = east, counter-clockwise).
        radius: wedge radius in data units.
        view_angle_rad: half-angle of the FOV in radians.
        color: fill colour.
        alpha: fill transparency.
    """
    theta1 = np.degrees(heading - view_angle_rad)
    theta2 = np.degrees(heading + view_angle_rad)
    wedge = mpatches.Wedge(
        center=(float(pos[0]), float(pos[1])),
        r=radius,
        theta1=theta1,
        theta2=theta2,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(wedge)


def _draw_contact_circle(
    ax: plt.Axes,
    pos: np.ndarray,
    radius: float,
    color: str,
) -> None:
    circle = mpatches.Circle(
        (float(pos[0]), float(pos[1])),
        radius=radius,
        fill=False,
        edgecolor=color,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        zorder=2,
    )
    ax.add_patch(circle)


# ── main class ────────────────────────────────────────────────────────────────


class EnhancedSearchAndRescueViewer:
    """Viewer that shows the full agent logic chain per frame.

    Each frame contains:
      • Main map: environment state with FOV cones, trajectories, HUD
      • Per-agent obs radar: polar plot of the 3-channel ray observation
      • Per-agent action bars: rotation & acceleration bars + result readout
    """

    def __init__(
        self,
        env_size: Tuple[float, float] = (1.0, 1.0),
        view_angle: float = 0.4,
        searcher_vision_range: float = 0.4,
        target_vision_range: float = 0.1,
        target_contact_range: float = 0.02,
        max_rotate: float = 0.25,
        trail_length: int = 20,
    ) -> None:
        self.env_size = env_size
        self.view_angle = view_angle  # fraction of π (half-angle)
        self.searcher_vision_range = searcher_vision_range
        self.target_vision_range = target_vision_range
        self.target_contact_range = target_contact_range
        self.max_rotate = max_rotate  # fraction of π
        self.trail_length = trail_length

    # ── public API ────────────────────────────────────────────────────────────

    def animate(
        self,
        states: Sequence[State],
        observations: Sequence[Observation],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an enhanced animation.

        Args:
            states: sequence of State objects (length T).
            observations: sequence of Observation objects (length T).
            actions: sequence of action arrays shape (num_agents, 2) (length T).
            rewards: sequence of reward arrays shape (num_agents,) (length T).
            interval: delay between frames in ms.
            save_path: if given, save the animation to this path.

        Returns:
            FuncAnimation object.
        """
        if not states:
            raise ValueError("states must be non-empty")

        num_agents = int(np.array(states[0].searchers.pos).shape[0])
        num_targets = int(np.array(states[0].targets.pos).shape[0])

        # ── figure layout ─────────────────────────────────────────────────────
        ncols = num_agents * 2
        fig_w = max(10, 5 + 3 * num_agents)
        fig_h = 10
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1A1A2E")

        gs = GridSpec(
            2,
            ncols,
            figure=fig,
            height_ratios=[3, 2],
            hspace=0.35,
            wspace=0.3,
            left=0.05,
            right=0.97,
            top=0.95,
            bottom=0.04,
        )

        # Main map
        main_ax = fig.add_subplot(gs[0, :])
        main_ax.set_facecolor("#0F0F23")
        main_ax.set_xlim(0, self.env_size[0])
        main_ax.set_ylim(0, self.env_size[1])
        main_ax.set_xticks([])
        main_ax.set_yticks([])
        for spine in main_ax.spines.values():
            spine.set_edgecolor("#444466")

        # Per-agent sub-axes
        obs_axes: List[plt.Axes] = []
        act_axes: List[plt.Axes] = []
        for i in range(num_agents):
            obs_ax = fig.add_subplot(gs[1, i * 2], projection="polar")
            act_ax = fig.add_subplot(gs[1, i * 2 + 1])
            obs_ax.set_facecolor("#0F0F23")
            act_ax.set_facecolor("#0F0F23")
            obs_axes.append(obs_ax)
            act_axes.append(act_ax)

        # Pre-convert all states to numpy for speed
        all_pos = [np.array(s.searchers.pos) for s in states]
        all_heading = [np.array(s.searchers.heading) for s in states]
        all_speed = [np.array(s.searchers.speed) for s in states]
        all_tpos = [np.array(s.targets.pos) for s in states]
        all_tfound = [np.array(s.targets.found) for s in states]
        all_step = [int(np.array(s.step)) for s in states]
        all_views = [np.array(o.searcher_views) for o in observations]
        all_actions = [np.array(a) for a in actions]
        all_rewards = [np.array(r) for r in rewards]

        view_angle_rad = self.view_angle * np.pi  # half-angle in radians

        # ── frame function ────────────────────────────────────────────────────

        def make_frame(frame_idx: int) -> None:
            pos = all_pos[frame_idx]  # (num_agents, 2)
            heading = all_heading[frame_idx]  # (num_agents,)
            speed = all_speed[frame_idx]  # (num_agents,)
            tpos = all_tpos[frame_idx]  # (num_targets, 2)
            tfound = all_tfound[frame_idx]  # (num_targets,) bool
            step = all_step[frame_idx]
            views = all_views[frame_idx]  # (num_agents, 3, 128)
            act = all_actions[frame_idx]  # (num_agents, 2)
            rew = all_rewards[frame_idx]  # (num_agents,)

            prev_found = all_tfound[max(0, frame_idx - 1)]
            newly_found = tfound & ~prev_found  # targets found THIS frame

            # ── main map ──────────────────────────────────────────────────────
            main_ax.cla()
            main_ax.set_facecolor("#0F0F23")
            main_ax.set_xlim(0, self.env_size[0])
            main_ax.set_ylim(0, self.env_size[1])
            main_ax.set_xticks([])
            main_ax.set_yticks([])

            # Targets
            for t_idx in range(num_targets):
                color = _TARGET_FOUND if tfound[t_idx] else _TARGET_UNFOUND
                main_ax.scatter(
                    tpos[t_idx, 0],
                    tpos[t_idx, 1],
                    c=color,
                    s=25,
                    zorder=3,
                    linewidths=0,
                )
                if newly_found[t_idx]:
                    pulse = mpatches.Circle(
                        (tpos[t_idx, 0], tpos[t_idx, 1]),
                        radius=0.03,
                        fill=False,
                        edgecolor=_TARGET_UNFOUND,
                        linewidth=2.0,
                        alpha=0.7,
                        zorder=4,
                    )
                    main_ax.add_patch(pulse)

            # Per-agent elements
            for i in range(num_agents):
                color = _agent_color(i)
                p = pos[i]
                h = float(heading[i])

                # Trajectory trail
                start = max(0, frame_idx - self.trail_length)
                trail_pos = np.array(all_pos[start : frame_idx + 1])  # (T, 2)
                if len(trail_pos) > 1:
                    n = len(trail_pos)
                    r_c = int(color[1:3], 16) / 255
                    g_c = int(color[3:5], 16) / 255
                    b_c = int(color[5:7], 16) / 255
                    for k in range(n - 1):
                        alpha = 0.05 + 0.55 * (k / (n - 1))
                        main_ax.plot(
                            trail_pos[k : k + 2, i, 0],
                            trail_pos[k : k + 2, i, 1],
                            color=(r_c, g_c, b_c, alpha),
                            linewidth=1.2,
                            zorder=2,
                        )

                # FOV outer wedge (searcher vision range)
                _draw_fov_wedge(
                    main_ax,
                    p,
                    h,
                    radius=self.searcher_vision_range,
                    view_angle_rad=view_angle_rad,
                    color=color,
                    alpha=0.07,
                )
                # FOV inner wedge (target vision range)
                _draw_fov_wedge(
                    main_ax,
                    p,
                    h,
                    radius=self.target_vision_range,
                    view_angle_rad=view_angle_rad,
                    color=color,
                    alpha=0.18,
                )
                # Contact circle
                _draw_contact_circle(main_ax, p, self.target_contact_range, color)

                # Agent arrow
                main_ax.quiver(
                    p[0],
                    p[1],
                    np.cos(h),
                    np.sin(h),
                    color=color,
                    pivot="middle",
                    width=0.006,
                    headwidth=5,
                    headlength=8,
                    headaxislength=8,
                    scale=18,
                    zorder=5,
                )

            # HUD
            found_count = int(tfound.sum())
            hud_text = f"Step: {step}   Targets: {found_count}/{num_targets}"
            main_ax.text(
                0.01,
                0.98,
                hud_text,
                transform=main_ax.transAxes,
                fontsize=9,
                color="white",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.6),
                zorder=10,
            )

            # ── per-agent panels ──────────────────────────────────────────────
            for i in range(num_agents):
                color = _agent_color(i)
                _draw_obs_radar(
                    obs_axes[i],
                    views[i],  # (3, 128)
                    view_angle_rad,
                    agent_idx=i,
                    color=color,
                )
                _draw_action_bars(
                    act_axes[i],
                    act[i],  # (2,)
                    rew[i],
                    float(speed[i]),
                    self.max_rotate,
                    agent_idx=i,
                    color=color,
                )

        # ── build animation ───────────────────────────────────────────────────
        anim = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=range(len(states)),
            interval=interval,
            blit=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=max(1, 1000 // interval))

        plt.close(fig)
        return anim


# ── panel drawing helpers ─────────────────────────────────────────────────────


def _draw_obs_radar(
    ax: plt.Axes,
    views: np.ndarray,  # (3, 128)
    view_angle_rad: float,
    agent_idx: int,
    color: str,
) -> None:
    """Draw a polar radar showing the 3-channel ray observation."""
    ax.cla()
    ax.set_facecolor("#0F0F23")

    num_rays = views.shape[1]
    # Ray angles: from -view_angle_rad to +view_angle_rad
    # In polar axes: theta=0 is east; we want 0 = forward (north-ish).
    # We'll keep theta=0 as the centre of the FOV and use raw angles.
    thetas = np.linspace(-view_angle_rad, view_angle_rad, num_rays)

    channel_colors = [_CH0_COLOR, _CH1_COLOR, _CH2_COLOR]
    channel_labels = ["agents", "found", "unfound"]

    for ch in range(3):
        raw = views[ch]  # (128,) values in [-1, 1], -1 = empty
        # Replace -1 (empty) with 0 for display
        r = np.where(raw < 0, 0.0, raw)
        # Close the polygon: repeat first point
        theta_plot = np.concatenate([thetas, [thetas[-1], thetas[0]]])
        r_plot = np.concatenate([r, [0.0, 0.0]])
        ax.fill(theta_plot, r_plot, color=channel_colors[ch], alpha=0.55, label=channel_labels[ch])
        ax.plot(thetas, r, color=channel_colors[ch], linewidth=0.8, alpha=0.9)

    # Restrict visible angle range
    deg_min = np.degrees(-view_angle_rad)
    deg_max = np.degrees(view_angle_rad)
    ax.set_thetamin(deg_min)
    ax.set_thetamax(deg_max)
    ax.set_rlim(0, 1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=5, colors="#888888")
    ax.set_theta_zero_location("N")  # 0 rad = top (forward)
    ax.set_theta_direction(-1)  # clockwise = right

    # Grid styling
    ax.grid(color="#333355", linewidth=0.5, alpha=0.6)
    ax.spines["polar"].set_color("#333355")

    # Title
    r_c = int(color[1:3], 16) / 255
    g_c = int(color[3:5], 16) / 255
    b_c = int(color[5:7], 16) / 255
    ax.set_title(
        f"Agent {agent_idx}  OBS",
        fontsize=8,
        color=color,
        pad=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(r_c, g_c, b_c, 0.15)),
    )

    # Legend (tiny)
    legend_patches = [
        mpatches.Patch(color=_CH0_COLOR, label="agents"),
        mpatches.Patch(color=_CH1_COLOR, label="found"),
        mpatches.Patch(color=_CH2_COLOR, label="unfound"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=5,
        framealpha=0.0,
        labelcolor="#AAAAAA",
    )


def _draw_action_bars(
    ax: plt.Axes,
    action: np.ndarray,  # (2,)
    reward: float,
    speed: float,
    max_rotate: float,
    agent_idx: int,
    color: str,
) -> None:
    """Draw action bars (rotation, acceleration) and result readout."""
    ax.cla()
    ax.set_facecolor("#0F0F23")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    rot_val = float(action[0])  # [-1, 1]
    acc_val = float(action[1])  # [-1, 1]

    bar_height = 0.35
    bar_y_rot = 2.7
    bar_y_acc = 1.8

    # ── background bars ───────────────────────────────────────────────────────
    for bar_y in [bar_y_rot, bar_y_acc]:
        ax.barh(
            bar_y,
            2.0,
            left=-1.0,
            height=bar_height,
            color="#2A2A4A",
            zorder=1,
        )

    # ── centre line ───────────────────────────────────────────────────────────
    for bar_y in [bar_y_rot, bar_y_acc]:
        ax.axvline(
            0,
            ymin=(bar_y - bar_height / 2) / 4,
            ymax=(bar_y + bar_height / 2) / 4,
            color="#666688",
            linewidth=0.8,
            zorder=2,
        )

    # ── rotation bar ──────────────────────────────────────────────────────────
    ax.barh(
        bar_y_rot,
        rot_val,
        left=0.0,
        height=bar_height,
        color=_ROT_COLOR,
        alpha=0.85,
        zorder=3,
    )
    ax.text(-1.25, bar_y_rot, "rot", va="center", ha="left", fontsize=7, color="#AAAACC")
    rot_deg = rot_val * max_rotate * 180
    ax.text(
        1.25, bar_y_rot, f"{rot_deg:+.1f}°", va="center", ha="right", fontsize=7, color=_ROT_COLOR
    )

    # ── acceleration bar ──────────────────────────────────────────────────────
    ax.barh(
        bar_y_acc,
        acc_val,
        left=0.0,
        height=bar_height,
        color=_ACC_COLOR,
        alpha=0.85,
        zorder=3,
    )
    ax.text(-1.25, bar_y_acc, "acc", va="center", ha="left", fontsize=7, color="#AAAACC")
    ax.text(
        1.25, bar_y_acc, f"{acc_val:+.2f}", va="center", ha="right", fontsize=7, color=_ACC_COLOR
    )

    # ── x-axis ticks ─────────────────────────────────────────────────────────
    for xv, lbl in [(-1, "-1"), (0, "0"), (1, "+1")]:
        ax.text(xv, 1.35, lbl, va="top", ha="center", fontsize=6, color="#666688")

    # ── result readout ────────────────────────────────────────────────────────
    # Speed bar (normalised 0→1 between min and max speed)
    speed_min, speed_max = 0.005, 0.02
    speed_norm = (speed - speed_min) / (speed_max - speed_min)
    ax.barh(0.75, 1.0, left=0.0, height=0.28, color="#2A2A4A", zorder=1)
    ax.barh(0.75, speed_norm, left=0.0, height=0.28, color=color, alpha=0.75, zorder=2)
    ax.text(-1.25, 0.75, "spd", va="center", ha="left", fontsize=7, color="#AAAACC")
    ax.text(1.25, 0.75, f"{speed:.4f}", va="center", ha="right", fontsize=7, color=color)

    # Reward
    rew_color = _TARGET_UNFOUND if float(reward) > 0 else "#666688"
    rew_text = f"reward: {float(reward):.3f}"
    ax.text(
        0.0,
        0.15,
        rew_text,
        va="center",
        ha="center",
        fontsize=8,
        color=rew_color,
        fontweight="bold" if float(reward) > 0 else "normal",
    )

    # Title
    r_c = int(color[1:3], 16) / 255
    g_c = int(color[3:5], 16) / 255
    b_c = int(color[5:7], 16) / 255
    ax.set_title(
        f"Agent {agent_idx}  ACTION",
        fontsize=8,
        color=color,
        pad=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(r_c, g_c, b_c, 0.15)),
    )
