"""Floating green-circle panel.

The window opens on its own, in the same way an RViz panel pops up.
Its tab lists only the green floor circles already in the warehouse,
numbered 1, 2, 3, and so on. This module does not train and does not
open a browser page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_tabs.catalog import checkpoint_for_number, numbered_checkpoints
from checkpoint_tabs.selector import CheckpointSelector
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311"
DEFAULT_OUTPUT = ROOT / "artifacts/warehouse_ppo/checkpoint_tabs"


class CheckpointPopup:
    """Numbered checkpoint buttons bound to one selector."""

    def __init__(self, selector: CheckpointSelector):
        self.selector = selector
        self.points = numbered_checkpoints()
        self._notice = ""

    def press(self, number: int) -> dict:
        """Drive to the checkpoint for this button number."""

        point = checkpoint_for_number(number)
        try:
            status = self.selector.start(point["id"])
        except SystemExit as exc:
            self._notice = str(exc)
            raise
        self._notice = ""
        return status

    def show(self) -> int:
        """Open the panel and keep it above the other windows until it is closed."""

        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Green circles")
        root.geometry("360x560+40+80")
        root.attributes("-topmost", True)
        root.configure(background="#383838")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#383838")
        style.configure("Panel.TLabel", background="#383838", foreground="#f0f0f0")
        style.configure("Title.TLabel", background="#383838", foreground="#f0f0f0", font=("sans-serif", 12, "bold"))
        style.configure(
            "Number.TButton",
            font=("sans-serif", 12, "bold"),
            padding=4,
        )

        frame = ttk.Frame(root, style="Panel.TFrame", padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Green circles", style="Title.TLabel").pack(anchor="w")

        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True, pady=(8, 8))
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Green circles")

        buttons = []
        for point in self.points:
            row = ttk.Frame(tab)
            row.pack(fill="x", pady=2)
            button = ttk.Button(
                row,
                text=str(point["number"]),
                width=4,
                style="Number.TButton",
                command=lambda number=point["number"]: self._on_press(number),
            )
            button.pack(side="left")
            ttk.Label(row, text=point["label"]).pack(side="left", padx=8)
            buttons.append(button)

        status = ttk.Label(frame, text="Waiting for a checkpoint.", style="Panel.TLabel", wraplength=330)
        status.pack(anchor="w")

        def refresh() -> None:
            current = self.selector.status()
            driving = current["state"] == "driving"
            for button in buttons:
                button.configure(state="disabled" if driving else "normal")
            if self._notice:
                status.configure(text=self._notice)
            elif current["state"] == "driving":
                status.configure(text=f"Driving to {current['checkpoint_id']}.")
            elif current["state"] == "finished":
                result = current.get("result") or {}
                distance = result.get("goal_distance")
                extra = "" if distance is None else f" Final distance {float(distance):.2f} m."
                status.configure(
                    text=f"{current['checkpoint_id']} finished: {result.get('termination_reason', 'done')}.{extra}"
                )
            elif current["state"] == "error":
                status.configure(text=current.get("error") or "Navigation failed.")
            else:
                status.configure(text="Waiting for a checkpoint.")
            root.after(200, refresh)

        root.after(200, refresh)
        root.mainloop()
        return 0

    def _on_press(self, number: int) -> None:
        try:
            self.press(number)
        except (SystemExit, RuntimeError, KeyError) as exc:
            self._notice = str(exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a popup tab for the existing green circles. Buttons 1, 2, 3, ... send the Jackal to that circle.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs/warehouse_ppo.yaml"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    config = load_warehouse_config(args.config)
    selector = CheckpointSelector(
        config,
        Path(args.checkpoint),
        authorize_live=live_authorized(args.authorize_live),
        seed=args.seed,
        output_dir=Path(args.output),
    )
    return CheckpointPopup(selector).show()


if __name__ == "__main__":
    raise SystemExit(main())
