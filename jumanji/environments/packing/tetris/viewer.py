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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.packing.tetris.constants import TETROMINOES_LIST
from jumanji.environments.packing.tetris.types import State
from jumanji.viewer import MatplotlibViewer

_DROP_EDGE_ROWS = 2


class TetrisViewer(MatplotlibViewer[State]):
    def __init__(self, num_rows: int, num_cols: int, render_mode: str = "human") -> None:
        """
        Viewer for a `Tetris` environment.

        Args:
            num_rows: Number of environment rows
            num_cols: Number of environment columns
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.n_colors = 10
        self._all_tetrominoes = jnp.array(TETROMINOES_LIST, jnp.int32)  # (7, 4, 4, 4)

        # Pick colors.
        colormap_indicies = jnp.arange(0, 1, 1 / self.n_colors)
        colormap = plt.get_cmap("hsv", self.n_colors + 1)

        self.colors = [(1.0, 1.0, 1.0, 1.0)]  # Initial color must be white.
        for colormap_idx in colormap_indicies:
            self.colors.append(colormap(colormap_idx))
        self.edgecolors = [(0.0, 0.0, 0.0), (0.9, 0.9, 0.9)]

        super().__init__(f"{num_rows}x{num_cols} Tetris", render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render Tetris.

        Args:
            state: State of the Tetris environment to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        fig.suptitle(f"Tetris    Score: {int(state.score)}", size=20)
        ax.invert_yaxis()
        grid = self._create_rendering_grid(state)
        self._add_grid_image(ax, grid)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def _move_tetromino(self, state: State, old_padded_grid: chex.Array) -> List[chex.Array]:
        """Shifts the tetromino from center to the selected position.

        Args:
            state: `State` object containing the current environment state.
            old_padded_grid: `chex.Array` containing the grid before placing the tetromino.

        Returns:
            grids: `List[NDArray]` contais a list of grids.
        """
        grids = []
        grid = old_padded_grid[: self.num_rows, : self.num_cols]
        center_position = self.num_cols - 4
        # step is 1 to move to the right and -1 to move to the left
        step = 1 if center_position < state.x_position else -1
        for xi in range(center_position, state.x_position + step, step):
            tetromino_zonne = jnp.zeros((4, state.grid_padded.shape[1]))
            tetromino_zonne = tetromino_zonne.at[0:4, xi : xi + 4].add(state.old_tetromino_rotated)
            # Delete the cols dedicated for the right padding
            tetromino_zonne = tetromino_zonne[:, : self.num_cols]
            # Stack the tetromino with grid position
            mixed_grid = jnp.vstack((tetromino_zonne, grid))
            grids.append(mixed_grid)
        return grids

    def _crush_lines(self, state: State, grid: chex.Array, n: int = 2) -> List[chex.Array]:
        """Creates animation when a line is crushed by toggling its value.

        Args:
            state: `State` object containing the current environment state.
            grid: `chex.Array` (self.num_rows, self.num_cols)
            n: `int`, optional, defines the number of repetitions. Defaults to 2.

        Returns:
            List[chex.Array]: Sequence of grids.
        """
        full_lines = state.full_lines[: self.num_rows]  # (num_rows,)
        full_lines_col = full_lines[:, jnp.newaxis]  # (num_rows, 1)
        cleared = jnp.where(~full_lines_col, grid, jnp.zeros((1, grid.shape[1])))
        result = []
        for _ in range(n):
            result.append(grid)
            result.append(cleared)
        return result

    def _create_rendering_grid(self, state: State) -> chex.Array:
        """Create a grid that contains tetromino and the envirement gerid.

        Args:
            state: `State` object containing the current environment state.

        Returns:
            rendering_grid: `chex.Array` (self.num_rows+4, self.num_cols)
        """
        grid = state.grid_padded[: self.num_rows, : self.num_cols]
        tetromino = jnp.zeros((4, self.num_cols))
        center_position = self.num_cols - 4
        tetromino_color_id = state.grid_padded.max() + 1
        colored_tetromino = state.new_tetromino * tetromino_color_id
        tetromino = tetromino.at[0:4, center_position : center_position + 4].set(colored_tetromino)
        rendering_grid = jnp.vstack((tetromino, grid))
        return rendering_grid

    def _drop_tetromino(self, state: State, old_padded_grid: chex.Array) -> List[NDArray]:
        """Creates animation while the tetromino is droping verticaly.

        Args:
            state: `State` object containing the current environment state.
            old_padded_grid: `chex.Array` containing the grid before placing the last tetromino

        Returns:
            grids: List[NDArray] contais a list of grids.
        """
        grids = []
        # `y_position` describes the position of the tetromino in the grid.
        # `y_position` may contain a value -1 if it bellongs to first tetromino.
        y_position = state.y_position if state.y_position != -1 else self.num_rows - 1
        # Stack the tetromino's rows on top of the grid.
        rendering_grid = jnp.vstack((jnp.zeros((4, old_padded_grid.shape[1])), old_padded_grid))
        # the animation grid contains 4 rows at the top dedicated to show the tetromino.
        for yi in range(y_position + 4 + 1):
            # Place the tetromino.
            grid = rendering_grid.at[yi : yi + 4, state.x_position : state.x_position + 4].add(
                state.old_tetromino_rotated
            )
            # Crop the grid (delete the 3 rows and columns padding at the bottom and the right.)
            grid = grid[: self.num_rows + 4, : self.num_cols]
            grids.append(grid)
        return grids

    def _drop_tetromino_simple(self, state: State, old_padded_grid: chex.Array) -> List[chex.Array]:
        """Drop animation showing only top ≤2 and bottom ≤2 frames.

        Args:
            state: `State` object containing the current environment state.
            old_padded_grid: `chex.Array` (num_rows+3, num_cols+3) before placing the tetromino.

        Returns:
            List[chex.Array]: Sequence of grids of shape (num_rows, num_cols).
        """
        y_position = int(state.y_position)
        if y_position == -1:
            y_position = self.num_rows - 1
        x_position = int(state.x_position)
        tetromino = state.old_tetromino_rotated  # (4, 4), colour-coded

        def _frame_at(yi: int) -> chex.Array:
            row_start = max(yi, 0)
            row_end = min(yi + 4, self.num_rows)
            tet_row_start = row_start - yi
            g = old_padded_grid.at[
                row_start:row_end,
                x_position : x_position + 4,
            ].add(tetromino[tet_row_start : tet_row_start + (row_end - row_start), :])
            return g[: self.num_rows, : self.num_cols]

        all_yi = list(range(0, y_position + 1))
        total = len(all_yi)
        edge = _DROP_EDGE_ROWS
        if total <= 2 * edge:
            selected_yi = all_yi
        else:
            selected_yi = all_yi[:edge] + all_yi[total - edge :]
        return [_frame_at(yi) for yi in selected_yi]

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of Tetris states.

        Changes from original:
        - Horizontal move phase removed; tetromino starts at final x_position.
        - Drop shows only top ≤2 and bottom ≤2 frames (edge-only).
        - Rotation-preview panel above main grid shows all 4 rotations of the
          just-placed piece with the chosen rotation highlighted.
        """
        fig_name = self._name + "_animation"
        fig = plt.figure(fig_name, figsize=self.figure_size)
        plt.close(fig=fig)

        gs = gridspec.GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[1, self.num_rows],
            hspace=0.15,
        )
        ax_panel = fig.add_subplot(gs[0])
        ax_main = fig.add_subplot(gs[1])

        frames: List[Tuple[chex.Array, float, chex.Array, int, int]] = []

        for state in states:
            if state.is_reset:
                continue
            score = float(state.score - state.reward)
            type_idx, rot_idx = self._infer_placed_piece(state)
            all_rotations = self._all_tetrominoes[type_idx]  # (4, 4, 4)

            drop_grids = self._drop_tetromino_simple(state, state.grid_padded_old)
            for g in drop_grids:
                frames.append((g, score, all_rotations, rot_idx, type_idx))

            if state.full_lines[: self.num_rows].sum() > 0:
                crush_grids = self._crush_lines(state, drop_grids[-1])
                for g in crush_grids:
                    frames.append((g, score, all_rotations, rot_idx, type_idx))

        if not frames:
            frames.append(
                (
                    jnp.zeros((self.num_rows, self.num_cols), jnp.int32),
                    0.0,
                    self._all_tetrominoes[0],
                    0,
                    0,
                )
            )

        def make_frame(
            frame_data: Tuple[chex.Array, float, chex.Array, int, int],
        ) -> Tuple[Artist, ...]:
            grid, score, all_rots, sel_rot, t_idx = frame_data
            ax_main.clear()
            ax_main.invert_yaxis()
            self._add_grid_image(ax_main, grid, is_animate=True)

            ax_panel.clear()
            self._draw_rotation_panel(ax_panel, all_rots, sel_rot, t_idx)

            fig.suptitle(f"Tetris    Score: {int(score)}", size=20)
            return (ax_main, ax_panel)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=frames,
            interval=interval,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _infer_placed_piece(self, state: State) -> Tuple[int, int]:
        """Return (tetromino_type_idx, rotation_idx) for the just-placed piece.

        Scans all 7 types x 4 rotations. Returns lowest-index match.
        Falls back to (0, 0) if no match (e.g. reset state with all-zeros).
        """
        placed_binary = (state.old_tetromino_rotated > 0).astype(jnp.int32)
        for t in range(self._all_tetrominoes.shape[0]):
            for r in range(self._all_tetrominoes.shape[1]):
                if jnp.array_equal(placed_binary, self._all_tetrominoes[t, r]):
                    return t, r
        return 0, 0

    def _draw_rotation_panel(
        self,
        ax: plt.Axes,
        all_rotations: chex.Array,  # (4, 4, 4)
        selected_rot_idx: int,
        type_idx: int,
    ) -> None:
        ax.set_axis_off()
        gap = 1
        mini_w = 4
        total_w = 4 * mini_w + 3 * gap  # 19

        color_id = (type_idx % (len(self.colors) - 1)) + 1

        for r in range(4):
            tet = all_rotations[r]  # (4, 4)
            x_offset = r * (mini_w + gap)
            is_selected = r == selected_rot_idx

            for row in range(4):
                for col in range(4):
                    cell_val = int(tet[row, col])
                    if cell_val == 0:
                        fc = (0.93, 0.93, 0.93, 1.0)
                        ec = (0.80, 0.80, 0.80, 1.0)
                        lw = 0.5
                    elif is_selected:
                        fc = self.colors[color_id]
                        ec = (0.0, 0.0, 0.0, 1.0)
                        lw = 1.5
                    else:
                        fc = (0.6, 0.6, 0.6, 0.4)
                        ec = (0.5, 0.5, 0.5, 0.6)
                        lw = 0.5
                    rect = plt.Rectangle(
                        (x_offset + col, row),
                        1,
                        1,
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=lw,
                    )
                    ax.add_patch(rect)

            label = f"{r * 90}°"
            label_kwargs: Dict[str, Any] = {"ha": "center", "va": "top", "fontsize": 7}
            if is_selected:
                label_kwargs["fontweight"] = "bold"
                label_kwargs["color"] = "black"
            else:
                label_kwargs["color"] = "grey"
            ax.text(x_offset + mini_w / 2, 4.3, label, **label_kwargs)

        sel_x = selected_rot_idx * (mini_w + gap)
        highlight = plt.Rectangle(
            (sel_x - 0.1, -0.1),
            mini_w + 0.2,
            4.2,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            linestyle="--",
        )
        ax.add_patch(highlight)

        ax.set_xlim(-0.5, total_w + 0.5)
        ax.set_ylim(5.0, -0.5)
        ax.set_aspect(1)

    def _add_grid_image(self, ax: plt.Axes, grid: chex.Array, is_animate: bool = False) -> None:
        self._draw_grid(grid, ax, is_animate=is_animate)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: chex.Array, ax: plt.Axes, is_animate: bool = False) -> None:
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                self._draw_grid_cell(grid[row, col], row, col, ax, is_animate=is_animate)

    def _draw_grid_cell(
        self,
        cell_value: int,
        row: int,
        col: int,
        ax: plt.Axes,
        is_animate: bool = False,
    ) -> None:
        # render() mode: rows 0-3 are the virtual preview zone → dashed borders.
        # animate() mode: grid is cropped to real (num_rows, num_cols) → all solid.
        is_padd = False if is_animate else row < 4
        cell = plt.Rectangle((col, row), 1, 1, **self._get_cell_attributes(cell_value, is_padd))
        ax.add_patch(cell)

    def _get_cell_attributes(self, cell_value: int, is_padd: bool) -> Dict[str, Any]:
        cell_value = int(cell_value)
        color_id = cell_value if cell_value == 0 else cell_value % (len(self.colors) - 1) + 1

        color = self.colors[color_id]
        edge_color = self.edgecolors[is_padd]
        return {"facecolor": color, "edgecolor": edge_color, "linewidth": 1}
