from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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
_CH0_COLOR = "#5B9BD5"  # other agents in obs
_CH1_COLOR = "#888888"  # found targets in obs
_CH2_COLOR = "#E98449"  # unfound targets in obs
_ROT_COLOR = "#9B59B6"  # rotation action bar
_ACC_COLOR = "#27AE60"  # acceleration action bar

_BG_COLOR = "white"
_PANEL_BG = "#F5F5F5"
_TEXT_COLOR = "#222222"
_GRID_COLOR = "#CCCCCC"
_LABEL_COLOR = "#555555"


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
        target_visual_radius: float = 0.02,
        max_rotate: float = 0.25,
        trail_length: int = 20,
    ) -> None:
        self.env_size = env_size
        self.view_angle = view_angle
        self.searcher_vision_range = searcher_vision_range
        self.target_vision_range = target_vision_range
        self.target_contact_range = target_contact_range
        self.target_visual_radius = target_visual_radius
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
        # Left column: square main map.
        # Right column: per-agent panels stacked vertically, each row = [obs | action].
        map_size_in = 7.0
        panel_row_h = 3.2  # height per agent row
        panel_col_w = 3.0  # width of obs panel
        action_col_w = 3.0  # width of action panel
        right_w = panel_col_w + action_col_w + 0.3  # total right-column width

        fig_h = max(map_size_in + 0.6, num_agents * panel_row_h + 0.6)
        fig_w = map_size_in + right_w + 0.7

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)

        margin_l = 0.03
        margin_r = 0.05
        margin_top = 0.03
        margin_bot = 0.04
        gap_lr = 0.04

        map_frac_w = map_size_in / fig_w
        map_frac_h = map_size_in / fig_h
        map_left = margin_l
        map_bottom = (1.0 - map_frac_h) / 2  # vertically centred

        main_ax = fig.add_axes(
            [map_left, map_bottom, map_frac_w, map_frac_h],
            aspect="equal",
        )
        main_ax.set_facecolor(_BG_COLOR)
        main_ax.set_xlim(0, self.env_size[0])
        main_ax.set_ylim(0, self.env_size[1])
        main_ax.set_xticks([])
        main_ax.set_yticks([])
        for spine in main_ax.spines.values():
            spine.set_edgecolor("#AAAAAA")

        # Right column: each agent gets one row split into [obs | action]
        right_start_x = map_left + map_frac_w + gap_lr
        right_total_w = 1.0 - margin_r - right_start_x
        obs_frac_w = right_total_w / 2 - 0.01
        act_start_x = right_start_x + obs_frac_w + 0.02
        act_frac_w = 1.0 - margin_r - act_start_x

        row_frac_h = (1.0 - margin_top - margin_bot) / num_agents
        inner_h = row_frac_h - 0.04

        obs_axes: List[plt.Axes] = []
        act_axes: List[plt.Axes] = []
        for i in range(num_agents):
            row_bottom = margin_bot + (num_agents - 1 - i) * row_frac_h + 0.02

            obs_ax = fig.add_axes(
                [right_start_x, row_bottom, obs_frac_w, inner_h],
                projection="polar",
            )
            act_ax = fig.add_axes(
                [act_start_x, row_bottom, act_frac_w, inner_h],
            )
            obs_ax.set_facecolor(_PANEL_BG)
            act_ax.set_facecolor(_PANEL_BG)
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
            main_ax.set_facecolor(_BG_COLOR)
            main_ax.set_xlim(0, self.env_size[0])
            main_ax.set_ylim(0, self.env_size[1])
            main_ax.set_xticks([])
            main_ax.set_yticks([])
            main_ax.set_aspect("equal")

            for t_idx in range(num_targets):
                color = _TARGET_FOUND if tfound[t_idx] else _TARGET_UNFOUND
                dot = mpatches.Circle(
                    (tpos[t_idx, 0], tpos[t_idx, 1]),
                    radius=self.target_visual_radius,
                    facecolor=color,
                    edgecolor="none",
                    zorder=3,
                )
                main_ax.add_patch(dot)
                if newly_found[t_idx]:
                    pulse = mpatches.Circle(
                        (tpos[t_idx, 0], tpos[t_idx, 1]),
                        radius=self.target_visual_radius * 2.0,
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

                # Trajectory trail — skip segments that cross a wrapped boundary
                start = max(0, frame_idx - self.trail_length)
                trail_pos = np.array(all_pos[start : frame_idx + 1])  # (T, num_agents, 2)
                if len(trail_pos) > 1:
                    n = len(trail_pos)
                    r_c = int(color[1:3], 16) / 255
                    g_c = int(color[3:5], 16) / 255
                    b_c = int(color[5:7], 16) / 255
                    half = self.env_size[0] / 2.0
                    for k in range(n - 1):
                        p0 = trail_pos[k, i]
                        p1 = trail_pos[k + 1, i]
                        if abs(p1[0] - p0[0]) > half or abs(p1[1] - p0[1]) > half:
                            continue  # wrapped boundary — do not draw
                        alpha = 0.05 + 0.55 * (k / (n - 1))
                        main_ax.plot(
                            [p0[0], p1[0]],
                            [p0[1], p1[1]],
                            color=(r_c, g_c, b_c, alpha),
                            linewidth=1.5,
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
                    width=0.003,
                    headwidth=4,
                    headlength=5,
                    headaxislength=5,
                    scale=35,
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
                fontsize=13,
                color=_TEXT_COLOR,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"
                ),
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
    ax.set_facecolor(_PANEL_BG)

    num_rays = views.shape[1]
    thetas = np.linspace(-view_angle_rad, view_angle_rad, num_rays)

    channel_colors = [_CH0_COLOR, _CH1_COLOR, _CH2_COLOR]
    channel_labels = ["agents", "found", "unfound"]

    for ch in range(3):
        raw = views[ch]
        r = np.where(raw < 0, 0.0, raw)
        theta_plot = np.concatenate([thetas, [thetas[-1], thetas[0]]])
        r_plot = np.concatenate([r, [0.0, 0.0]])
        ax.fill(theta_plot, r_plot, color=channel_colors[ch], alpha=0.55, label=channel_labels[ch])
        ax.plot(thetas, r, color=channel_colors[ch], linewidth=1.2, alpha=0.9)

    deg_min = np.degrees(-view_angle_rad)
    deg_max = np.degrees(view_angle_rad)
    ax.set_thetamin(deg_min)
    ax.set_thetamax(deg_max)
    ax.set_rlim(0, 1.05)
    ax.set_rticks([0.5, 1.0])
    ax.tick_params(labelsize=9, colors=_LABEL_COLOR)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.grid(color=_GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.spines["polar"].set_color(_GRID_COLOR)

    r_c = int(color[1:3], 16) / 255
    g_c = int(color[3:5], 16) / 255
    b_c = int(color[5:7], 16) / 255
    ax.set_title(
        f"Agent {agent_idx}  OBS",
        fontsize=12,
        color=color,
        pad=2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=(r_c, g_c, b_c, 0.12), edgecolor="none"),
    )

    legend_patches = [
        mpatches.Patch(color=_CH0_COLOR, label="agents"),
        mpatches.Patch(color=_CH1_COLOR, label="found"),
        mpatches.Patch(color=_CH2_COLOR, label="unfound"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=9,
        framealpha=0.0,
        labelcolor=_LABEL_COLOR,
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
    ax.set_facecolor(_PANEL_BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    rot_val = float(action[0])
    acc_val = float(action[1])

    bar_height = 0.38
    bar_y_rot = 2.7
    bar_y_acc = 1.8

    for bar_y in [bar_y_rot, bar_y_acc]:
        ax.barh(bar_y, 2.0, left=-1.0, height=bar_height, color="#DDDDDD", zorder=1)

    for bar_y in [bar_y_rot, bar_y_acc]:
        ax.axvline(
            0,
            ymin=(bar_y - bar_height / 2) / 4,
            ymax=(bar_y + bar_height / 2) / 4,
            color="#999999",
            linewidth=1.0,
            zorder=2,
        )

    ax.barh(bar_y_rot, rot_val, left=0.0, height=bar_height, color=_ROT_COLOR, alpha=0.85, zorder=3)
    ax.text(-1.25, bar_y_rot, "rot", va="center", ha="left", fontsize=11, color=_LABEL_COLOR)
    rot_deg = rot_val * max_rotate * 180
    ax.text(
        1.25, bar_y_rot, f"{rot_deg:+.1f}°", va="center", ha="right", fontsize=11, color=_ROT_COLOR
    )

    ax.barh(bar_y_acc, acc_val, left=0.0, height=bar_height, color=_ACC_COLOR, alpha=0.85, zorder=3)
    ax.text(-1.25, bar_y_acc, "acc", va="center", ha="left", fontsize=11, color=_LABEL_COLOR)
    ax.text(
        1.25, bar_y_acc, f"{acc_val:+.2f}", va="center", ha="right", fontsize=11, color=_ACC_COLOR
    )

    for xv, lbl in [(-1, "-1"), (0, "0"), (1, "+1")]:
        ax.text(xv, 1.35, lbl, va="top", ha="center", fontsize=9, color="#888888")

    speed_min, speed_max = 0.005, 0.02
    speed_norm = np.clip((speed - speed_min) / (speed_max - speed_min), 0.0, 1.0)
    ax.barh(0.75, 1.0, left=0.0, height=0.30, color="#DDDDDD", zorder=1)
    ax.barh(0.75, speed_norm, left=0.0, height=0.30, color=color, alpha=0.75, zorder=2)
    ax.text(-1.25, 0.75, "spd", va="center", ha="left", fontsize=11, color=_LABEL_COLOR)
    ax.text(1.25, 0.75, f"{speed:.4f}", va="center", ha="right", fontsize=11, color=color)

    rew_color = _TARGET_UNFOUND if float(reward) > 0 else "#888888"
    ax.text(
        0.0,
        0.10,
        f"reward: {float(reward):.3f}",
        va="center",
        ha="center",
        fontsize=12,
        color=rew_color,
        fontweight="bold" if float(reward) > 0 else "normal",
    )

    r_c = int(color[1:3], 16) / 255
    g_c = int(color[3:5], 16) / 255
    b_c = int(color[5:7], 16) / 255
    ax.set_title(
        f"Agent {agent_idx}  ACTION",
        fontsize=12,
        color=color,
        pad=6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=(r_c, g_c, b_c, 0.12), edgecolor="none"),
    )
