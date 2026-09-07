"""Critter Counter desktop GUI.

Plug in a card, pick it (or browse to a folder), hit Go. Runs the pipeline in a
background thread and reports progress back to the main thread via a queue, since
Tkinter widgets may only be touched from the main thread.
"""

from __future__ import annotations

import os
import queue
import string
import threading
import traceback
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox

import cards
import pipeline
from config import load_config, save_config
from pipeline import Event, run_pipeline

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

# Scratch space for the per-card intermediate JSON stages. Kept under the user's home
# rather than next to the app so it still works if the app is installed somewhere
# read-only, and so it survives replacing the app folder with a newer build.
WORK_DIR = Path.home() / ".critter_counter" / "work"


def detect_dcim_drives() -> list[Path]:
    drives = []
    for letter in string.ascii_uppercase:
        candidate = Path(f"{letter}:/DCIM")
        if candidate.is_dir():
            drives.append(candidate)
    return drives


class CritterCounterApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Critter Counter")
        self.geometry("520x540")
        self.resizable(False, False)

        self.config_data = load_config()
        self.selected_folder: Path | None = None
        self.card_identity = None
        self.progress_queue: queue.Queue = queue.Queue()
        self.run_in_progress = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.select_frame = ctk.CTkFrame(self)
        self.progress_frame = ctk.CTkFrame(self)
        self.results_frame = ctk.CTkFrame(self)
        self.error_frame = ctk.CTkFrame(self)
        self.settings_frame = ctk.CTkFrame(self)

        self._build_select_frame()
        self._build_progress_frame()
        self._build_results_frame()
        self._build_error_frame()
        self._build_settings_frame()

        self.show_select_frame()

    # ---- Frame: select ----------------------------------------------------

    def _build_select_frame(self) -> None:
        f = self.select_frame
        ctk.CTkLabel(f, text="Critter Counter", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(30, 5))
        ctk.CTkLabel(f, text="Plug in a camera card, pick it below, and hit Go.").pack(pady=(0, 20))

        self.drive_buttons_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.drive_buttons_frame.pack(pady=5, fill="x", padx=30)
        self._refresh_drive_buttons()

        ctk.CTkButton(f, text="Refresh", width=100, command=self._refresh_drive_buttons).pack(pady=(5, 15))
        ctk.CTkButton(f, text="Browse for a folder...", command=self._browse_folder).pack(pady=5)

        self.selected_label = ctk.CTkLabel(f, text="No folder selected", text_color="gray")
        self.selected_label.pack(pady=10)

        self.go_button = ctk.CTkButton(
            f, text="Go", font=ctk.CTkFont(size=18, weight="bold"), height=50,
            state="disabled", command=self._start_pipeline,
        )
        self.go_button.pack(pady=(15, 5), padx=40, fill="x")

        ctk.CTkButton(
            f, text="Settings", width=90, fg_color="transparent", border_width=1,
            command=self._show_settings,
        ).pack(pady=(0, 10))

    def _refresh_drive_buttons(self) -> None:
        for child in self.drive_buttons_frame.winfo_children():
            child.destroy()

        drives = detect_dcim_drives()
        if not drives:
            ctk.CTkLabel(self.drive_buttons_frame, text="No camera cards detected", text_color="gray").pack()
            return

        for drive in drives:
            ctk.CTkButton(
                self.drive_buttons_frame, text=f"Card: {drive}",
                command=lambda d=drive: self._select_folder(d),
            ).pack(pady=3, fill="x")

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Select a folder of photos")
        if chosen:
            self._select_folder(Path(chosen))

    def _select_folder(self, folder: Path) -> None:
        self.selected_folder = folder
        self.card_identity = None
        self.selected_label.configure(text="Reading card...", text_color="gray")
        self.go_button.configure(state="disabled")

        # Identifying the card stats every photo on it, which is slow enough on a card
        # reader to freeze the window, so it happens off the main thread.
        threading.Thread(target=self._identify_card, args=(folder,), daemon=True).start()

    def _identify_card(self, folder: Path) -> None:
        try:
            identity = cards.identify(folder)
            summary = cards.describe(identity)
        except Exception as exc:
            identity, summary = None, f"Could not read {folder}\n{exc}"
        self.after(0, self._show_card_identity, folder, identity, summary)

    def _show_card_identity(self, folder: Path, identity, summary: str) -> None:
        if self.selected_folder != folder:
            return  # a different card was picked while this one was being read
        self.card_identity = identity
        self.selected_label.configure(text=summary, text_color=("black", "white"))
        if identity is not None and identity.photo_count:
            self.go_button.configure(state="normal")

    def show_select_frame(self) -> None:
        self.progress_frame.pack_forget()
        self.results_frame.pack_forget()
        self.error_frame.pack_forget()
        self.settings_frame.pack_forget()
        self._refresh_drive_buttons()
        self.select_frame.pack(fill="both", expand=True)

    # ---- Frame: progress ----------------------------------------------------

    def _build_progress_frame(self) -> None:
        f = self.progress_frame
        ctk.CTkLabel(f, text="Working...", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(60, 20))
        self.progress_bar = ctk.CTkProgressBar(f, width=400)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)
        self.progress_status_label = ctk.CTkLabel(f, text="Starting...")
        self.progress_status_label.pack(pady=10)
        ctk.CTkLabel(
            f, text="This can take several hours for a full card.\nYou can leave this running in the background.",
            text_color="gray",
        ).pack(pady=20)

    def _start_pipeline(self) -> None:
        if self.selected_folder is None:
            return
        self.select_frame.pack_forget()
        self.progress_bar.set(0)
        self.progress_status_label.configure(text="Starting...")
        self.progress_frame.pack(fill="both", expand=True)
        self.run_in_progress = True

        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()
        self.after(200, self._poll_progress_queue)

    def _run_pipeline_thread(self) -> None:
        def on_progress(fraction: float, status: str) -> None:
            self.progress_queue.put(("progress", fraction, status))

        try:
            output_root = Path(self.config_data["output_folder"])
            session_folder, events = run_pipeline(
                self.selected_folder,
                output_root,
                WORK_DIR,
                on_progress=on_progress,
                confidence_threshold=self.config_data["species_confidence_threshold"],
                event_gap_seconds=self.config_data["event_gap_seconds"],
                country=self.config_data["country"],
                admin1_region=self.config_data["state"],
            )
            if self.card_identity is not None:
                cards.record_run(self.card_identity, session_folder)
            self.progress_queue.put(("done", session_folder, events))
        except pipeline.NoPhotosFound as exc:
            # Expected mistake (wrong folder picked), not a crash - no stack trace.
            self.progress_queue.put(("error", str(exc), None))
        except Exception:
            self.progress_queue.put(("error", traceback.format_exc(), None))

    def _poll_progress_queue(self) -> None:
        try:
            while True:
                message = self.progress_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, fraction, status = message
                    self.progress_bar.set(fraction)
                    self.progress_status_label.configure(text=status)
                elif kind == "done":
                    _, session_folder, events = message
                    self.run_in_progress = False
                    self._show_results(session_folder, events)
                    return
                elif kind == "error":
                    _, error_text, _ = message
                    self.run_in_progress = False
                    self._show_error(error_text)
                    return
        except queue.Empty:
            pass
        self.after(200, self._poll_progress_queue)

    # ---- Frame: results ----------------------------------------------------

    def _build_results_frame(self) -> None:
        f = self.results_frame
        ctk.CTkLabel(f, text="Done!", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(30, 10))
        self.results_summary_label = ctk.CTkLabel(f, text="", justify="left")
        self.results_summary_label.pack(pady=10, padx=30)

        self.open_photos_button = ctk.CTkButton(f, text="Open Photos Folder", command=self._open_photos)
        self.open_photos_button.pack(pady=8)
        self.open_report_button = ctk.CTkButton(f, text="Open Excel Report", command=self._open_report)
        self.open_report_button.pack(pady=8)

        ctk.CTkButton(f, text="Process Another Card", command=self.show_select_frame).pack(pady=(25, 10))

        self._result_session_folder: Path | None = None

    def _build_settings_frame(self) -> None:
        f = self.settings_frame
        ctk.CTkLabel(f, text="Settings", font=ctk.CTkFont(size=22, weight="bold")).pack(
            pady=(20, 12)
        )

        rows = ctk.CTkFrame(f, fg_color="transparent")
        rows.pack(padx=25, fill="x")

        ctk.CTkLabel(rows, text="Save photos to").grid(row=0, column=0, sticky="w", pady=6)
        self.output_entry = ctk.CTkEntry(rows, width=230)
        self.output_entry.grid(row=0, column=1, pady=6)
        ctk.CTkButton(rows, text="...", width=32, command=self._browse_output).grid(
            row=0, column=2, padx=(6, 0)
        )

        ctk.CTkLabel(rows, text="State (e.g. TX)").grid(row=1, column=0, sticky="w", pady=6)
        self.state_entry = ctk.CTkEntry(rows, width=230)
        self.state_entry.grid(row=1, column=1, pady=6)

        ctk.CTkLabel(rows, text="Species confidence").grid(row=2, column=0, sticky="w", pady=6)
        self.threshold_entry = ctk.CTkEntry(rows, width=230)
        self.threshold_entry.grid(row=2, column=1, pady=6)

        ctk.CTkLabel(rows, text="Burst gap (seconds)").grid(row=3, column=0, sticky="w", pady=6)
        self.gap_entry = ctk.CTkEntry(rows, width=230)
        self.gap_entry.grid(row=3, column=1, pady=6)

        ctk.CTkLabel(
            f,
            text="Setting your state helps the AI rule out species that don't live\n"
            "near you, so fewer photos end up labeled Unknown.",
            text_color="gray",
            justify="left",
        ).pack(pady=(12, 5), padx=25)

        self.settings_error_label = ctk.CTkLabel(f, text="", text_color="red")
        self.settings_error_label.pack()

        buttons = ctk.CTkFrame(f, fg_color="transparent")
        buttons.pack(pady=10)
        ctk.CTkButton(buttons, text="Save", command=self._save_settings).pack(
            side="left", padx=6
        )
        ctk.CTkButton(
            buttons, text="Cancel", fg_color="gray", command=self.show_select_frame
        ).pack(side="left", padx=6)

    def _browse_output(self) -> None:
        chosen = filedialog.askdirectory(title="Where should photos be saved?")
        if chosen:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, chosen)

    def _show_settings(self) -> None:
        for entry, key in (
            (self.output_entry, "output_folder"),
            (self.state_entry, "state"),
            (self.threshold_entry, "species_confidence_threshold"),
            (self.gap_entry, "event_gap_seconds"),
        ):
            entry.delete(0, "end")
            entry.insert(0, str(self.config_data[key]))
        self.settings_error_label.configure(text="")
        self.select_frame.pack_forget()
        self.settings_frame.pack(fill="both", expand=True)

    def _save_settings(self) -> None:
        try:
            threshold = float(self.threshold_entry.get())
            gap = int(self.gap_entry.get())
        except ValueError:
            self.settings_error_label.configure(
                text="Confidence must be a number (0-1) and gap a whole number."
            )
            return
        if not 0.0 <= threshold <= 1.0:
            self.settings_error_label.configure(text="Confidence must be between 0 and 1.")
            return
        if gap < 0:
            self.settings_error_label.configure(text="Burst gap cannot be negative.")
            return

        self.config_data.update(
            {
                "output_folder": self.output_entry.get().strip(),
                "state": self.state_entry.get().strip().upper(),
                "species_confidence_threshold": threshold,
                "event_gap_seconds": gap,
            }
        )
        save_config(self.config_data)
        self.settings_frame.pack_forget()
        self.show_select_frame()

    def _on_close(self) -> None:
        """Stops the model subprocess before exiting.

        Without this the detector keeps running in the background after the window is
        gone - hours of invisible CPU/GPU work the user has no way to stop.
        """

        if self.run_in_progress:
            keep_going = messagebox.askyesno(
                "Stop processing?",
                "Processing is still running. Stop it and close?\n\n"
                "Finished work is saved - starting this card again will pick up "
                "where it left off.",
            )
            if not keep_going:
                return
            pipeline.terminate_active_process()
        self.destroy()

    def _build_error_frame(self) -> None:
        f = self.error_frame
        ctk.CTkLabel(
            f, text="Something went wrong", font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(25, 5))
        ctk.CTkLabel(
            f, text="Nothing was lost - finished work is saved and will be reused on retry.",
            text_color="gray",
        ).pack(pady=(0, 10))
        self.error_detail = ctk.CTkTextbox(f, width=450, height=160, wrap="word")
        self.error_detail.pack(padx=20, pady=5)
        self.error_detail.configure(state="disabled")
        ctk.CTkButton(f, text="Back", command=self.show_select_frame).pack(pady=15)

    def _show_results(self, session_folder: Path, events: list[Event]) -> None:
        self._result_session_folder = session_folder

        species_counts: dict[str, int] = {}
        for event in events:
            for species in event.species_present:
                species_counts[species] = species_counts.get(species, 0) + 1

        total_photos = sum(
            1 for event in events for img in event.images if img.has_detection
        )

        if species_counts:
            lines = [f"  {species}: {count}" for species, count in sorted(species_counts.items())]
            summary = f"{total_photos} good photos saved, by species (event count):\n" + "\n".join(lines)
        else:
            summary = "No animals found in this batch."

        self.results_summary_label.configure(text=summary)
        self.progress_frame.pack_forget()
        self.results_frame.pack(fill="both", expand=True)

    def _open_photos(self) -> None:
        if self._result_session_folder is not None:
            os.startfile(self._result_session_folder)

    def _open_report(self) -> None:
        if self._result_session_folder is not None:
            os.startfile(self._result_session_folder / "report.xlsx")

    def _show_error(self, error_text: str) -> None:
        """Shows the failure in the window itself.

        This runs under pythonw.exe, which has no console, so anything written to
        stdout/stderr would vanish and the user would just see the app give up.
        """

        self.progress_frame.pack_forget()
        self.results_frame.pack_forget()
        self.error_detail.configure(state="normal")
        self.error_detail.delete("1.0", "end")
        self.error_detail.insert("1.0", error_text)
        self.error_detail.configure(state="disabled")
        self.error_frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    app = CritterCounterApp()
    app.mainloop()
