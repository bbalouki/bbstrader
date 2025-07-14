import os
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from loguru import logger
from PIL import Image, ImageTk

from bbstrader.metatrader.copier import TradeCopier


def resource_path(relative_path):
    """Get absolute path to resource"""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = Path(__file__).resolve().parent.parent.parent

    return base_path / relative_path


TITLE = "BBS Trade Copier"
ICON_PATH = resource_path("assets/bbstrader.ico")
LOGO_PATH = resource_path("assets/bbstrader_logo.png")


class TradeCopierApp(object):
    def __init__(self, root: tk.Tk):
        root.title(TITLE)
        root.geometry("1600x900")
        self.root = root
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.trade_copier = None

        self.main_frame = self.add_main_frame()
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(3, weight=1)

        self.set_style()
        self.add_logo_and_description()
        self.add_source_account_frame(self.main_frame)
        self.add_destination_accounts_frame(self.main_frame)
        self.add_copier_settings(self.main_frame)

    def set_style(self):
        self.style = ttk.Style()
        self.style.configure("Bold.TLabelframe.Label", font=("Segoe UI", 15, "bold"))

    def add_main_frame(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure the layout
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # --- visual/logo frame ---
        self.visual_frame = ttk.Frame(main_frame)
        self.visual_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        # --- opier settings---
        self.right_panel_frame = ttk.Frame(main_frame)
        self.right_panel_frame.grid(
            row=1, column=1, rowspan=2, padx=5, pady=5, sticky="nsew"
        )
        self.right_panel_frame.columnconfigure(0, weight=1)
        self.right_panel_frame.rowconfigure(0, weight=0)
        self.right_panel_frame.rowconfigure(1, weight=1)

        return main_frame

    def add_source_account_frame(self, main_frame):
        # --- Source Account ---
        source_frame = ttk.LabelFrame(
            main_frame, text="Source Account", style="Bold.TLabelframe"
        )
        source_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="Login").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.source_login_entry = ttk.Entry(source_frame)
        self.source_login_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(source_frame, text="Password").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.source_password_entry = ttk.Entry(source_frame, show="*")
        self.source_password_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(source_frame, text="Server").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.source_server_entry = ttk.Entry(source_frame)
        self.source_server_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(source_frame, text="Path").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.source_path_entry = ttk.Entry(source_frame)
        self.source_path_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        source_path_browse_button = ttk.Button(
            source_frame,
            text="Browse...",
            command=lambda: self.browse_path(self.source_path_entry),
        )
        source_path_browse_button.grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)

    def add_destination_accounts_frame(self, main_frame):
        # --- Destination Accounts Scrollable Area ---
        self.destinations_outer_frame = ttk.LabelFrame(
            main_frame, text="Destination Accounts", style="Bold.TLabelframe"
        )
        self.destinations_outer_frame.grid(
            row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        self.destinations_outer_frame.rowconfigure(0, weight=1)
        self.destinations_outer_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.destinations_outer_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self.destinations_outer_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame_for_destinations = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame_for_destinations, anchor="nw"
        )

        self.scrollable_frame_for_destinations.columnconfigure(0, weight=1)

        def configure_scroll_region(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def configure_canvas_window(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.scrollable_frame_for_destinations.bind(
            "<Configure>", configure_scroll_region
        )
        self.canvas.bind("<Configure>", configure_canvas_window)

        def _on_mousewheel(event):
            scroll_val = -1 * (event.delta // 120)
            self.canvas.yview_scroll(scroll_val, "units")

        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scrollable_frame_for_destinations.bind("<MouseWheel>", _on_mousewheel)

        self.destination_widgets = []
        self.current_scrollable_content_row = 0

        self.add_dest_button = ttk.Button(
            self.destinations_outer_frame,
            text="Add Destination",
            command=self.add_destination_account,
        )
        self.add_dest_button.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

        self.add_destination_account()

    def add_copier_settings(self, main_frame):
        # --- Copier Settings ---
        settings_frame = ttk.LabelFrame(
            self.right_panel_frame, text="Copier Settings", style="Bold.TLabelframe"
        )
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(settings_frame, text="Sleep Time (s)").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.sleeptime_entry = ttk.Entry(settings_frame)
        self.sleeptime_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.sleeptime_entry.insert(0, "0.1")

        ttk.Label(settings_frame, text="Start Time (HH:MM)").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.start_time_entry = ttk.Entry(settings_frame)
        self.start_time_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(settings_frame, text="End Time (HH:MM)").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.end_time_entry = ttk.Entry(settings_frame)
        self.end_time_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        # --- Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(
            controls_frame, text="Start Copier", command=self.start_copier
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            controls_frame,
            text="Stop Copier",
            command=self.stop_copier,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # --- Log Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Logs", style="Bold.TLabelframe")
        log_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky=(tk.W, tk.E, tk.N, tk.S),
        )

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state="disabled")

    def add_logo_and_description(self):
        image = Image.open(LOGO_PATH)
        image = image.resize((120, 120))
        self.logo_img = ImageTk.PhotoImage(image)
        logo_label = ttk.Label(self.visual_frame, image=self.logo_img)
        logo_label.pack(padx=10, pady=10)

        # Add custom title/info
        ttk.Label(self.visual_frame, text=TITLE, font=("Segoe UI", 20, "bold")).pack()
        ttk.Label(
            self.visual_frame,
            text="Fast | Reliable | Flexible",
            font=("Segoe UI", 10, "bold"),
        ).pack()

    def add_destination_account(self):
        dest_row = len(self.destination_widgets)
        dest_widgets = {}

        if dest_row > 0:
            sep = ttk.Separator(
                self.scrollable_frame_for_destinations, orient=tk.HORIZONTAL
            )
            sep.grid(
                row=self.current_scrollable_content_row,
                column=0,
                sticky=tk.EW,
                pady=(10, 5),
            )
            self.current_scrollable_content_row += 1

        frame = ttk.Frame(self.scrollable_frame_for_destinations)
        frame.grid(
            row=self.current_scrollable_content_row,
            column=0,
            pady=5,
            padx=5,
            sticky=tk.EW,
        )
        self.current_scrollable_content_row += 1

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        ttk.Label(
            frame, text=f"Destination {dest_row + 1}", font=("Segoe UI", 8, "bold")
        ).grid(row=0, column=0, columnspan=5, sticky=tk.W, padx=5, pady=(5, 10))

        # Row 1: Login and Password
        ttk.Label(frame, text="Login").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["login"] = ttk.Entry(frame)
        dest_widgets["login"].grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(frame, text="Password").grid(
            row=1, column=2, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["password"] = ttk.Entry(frame, show="*")
        dest_widgets["password"].grid(
            row=1, column=3, columnspan=2, sticky=tk.EW, padx=5, pady=2
        )

        # Row 2: Server and Path
        ttk.Label(frame, text="Server").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["server"] = ttk.Entry(frame)
        dest_widgets["server"].grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(frame, text="Path").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        dest_widgets["path"] = ttk.Entry(frame)
        dest_widgets["path"].grid(row=2, column=3, sticky=tk.EW, padx=5, pady=2)
        dest_path_browse_button = ttk.Button(
            frame,
            text="Browse...",
            command=lambda w=dest_widgets["path"]: self.browse_path(w),
        )
        dest_path_browse_button.grid(row=2, column=4, sticky=tk.W, padx=(0, 5), pady=2)

        # Row 3: Mode and Value
        ttk.Label(frame, text="Mode").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        dest_widgets["mode"] = ttk.Combobox(
            frame,
            values=["fix", "multiply", "percentage", "dynamic", "replicate"],
            width=10,
        )
        dest_widgets["mode"].grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        dest_widgets["mode"].set("fix")

        ttk.Label(frame, text="Value").grid(
            row=3, column=2, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["value"] = ttk.Entry(frame)
        dest_widgets["value"].grid(
            row=3, column=3, columnspan=2, sticky=tk.EW, padx=5, pady=2
        )
        dest_widgets["value"].insert(0, "0.01")

        # Row 4: Copy what and slippage
        ttk.Label(frame, text="Copy What").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["copy_what"] = ttk.Combobox(
            frame, values=["all", "orders", "positions"], width=10
        )
        dest_widgets["copy_what"].grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        dest_widgets["copy_what"].set("all")

        ttk.Label(frame, text="Slippage").grid(
            row=4, column=2, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["slippage"] = ttk.Entry(frame)
        dest_widgets["slippage"].grid(
            row=4, column=3, columnspan=2, sticky=tk.EW, padx=5, pady=2
        )
        dest_widgets["slippage"].insert(0, "0.1")

        # Row 5: Symbols
        ttk.Label(frame, text="Symbols").grid(
            row=5, column=0, sticky=tk.W, padx=5, pady=2
        )
        dest_widgets["symbols"] = ttk.Entry(frame)
        dest_widgets["symbols"].grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
        dest_widgets["symbols"].insert(0, "all")

        symbols_load_button = ttk.Button(
            frame,
            text="Load...",
            command=lambda w=dest_widgets["symbols"]: self.browse_symbols_file(w),
        )
        symbols_load_button.grid(row=5, column=2, sticky=tk.W, padx=(0, 5), pady=2)

        self.destination_widgets.append(dest_widgets)

        self.add_dest_button.grid_forget()
        self.add_dest_button.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

    def log_message(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def _handle_symbols(self, symbols_str: str):
        symbols = symbols_str.strip().replace("\n", "").replace('"""', "")
        if symbols in ["all", "*"]:
            return symbols
        elif ":" in symbols:
            symbols = dict(item.split(":") for item in symbols.split(","))
        elif " " in symbols and "," not in symbols:
            symbols = symbols.split()
        elif "," in symbols:
            symbols = symbols.replace(" ", "").split(",")
        return symbols

    def _validate_inputs(self):
        if (
            not self.source_login_entry.get()
            or not self.source_password_entry.get()
            or not self.source_server_entry.get()
            or not self.source_path_entry.get()
        ):
            messagebox.showerror("Error", "Source account details are incomplete.")
            return False

        for i, dest in enumerate(self.destination_widgets):
            if (
                not dest["login"].get()
                or not dest["password"].get()
                or not dest["server"].get()
                or not dest["path"].get()
            ):
                messagebox.showerror(
                    "Error", f"Destination account {i+1} details are incomplete."
                )
                return False
            if (
                dest["mode"].get() in ["fix", "multiply", "percentage"]
                and not dest["value"].get()
            ):
                messagebox.showerror(
                    "Error",
                    f"Value is required for mode '{dest['mode'].get()}' in Destination {i+1}.",
                )
                return False
            try:
                if dest["value"].get():
                    float(dest["value"].get())
                if dest["slippage"].get():
                    float(dest["slippage"].get())
            except ValueError:
                messagebox.showerror(
                    "Error",
                    f"Invalid Value or Slippage for Destination {i+1}. "
                    "Must be a number (Slippage must be an integer).",
                )
                return False
        try:
            float(self.sleeptime_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Sleep Time. Must be a number.")
            return False
        # Add more validation for time formats if needed
        return True

    def start_copier(self):
        if not self._validate_inputs():
            return

        def gui_safe_logger(msg):
            # Ensures GUI thread safety
            self.log_text.after(0, self.log_message(msg))

        logger.add(
            gui_safe_logger,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        )

        source_config = {
            "login": int(self.source_login_entry.get().strip()),
            "password": self.source_password_entry.get().strip(),
            "server": self.source_server_entry.get().strip(),
            "path": self.source_path_entry.get().strip(),
        }

        destinations_config = []
        for dest_widget_map in self.destination_widgets:
            symbols_str = dest_widget_map["symbols"].get().strip()
            dest = {
                "login": int(dest_widget_map["login"].get().strip()),
                "password": dest_widget_map["password"].get().strip(),
                "server": dest_widget_map["server"].get().strip(),
                "path": dest_widget_map["path"].get().strip(),
                "mode": dest_widget_map["mode"].get().strip(),
                "symbols": self._handle_symbols(symbols_str),
                "copy_what": dest_widget_map["copy_what"].get().strip(),
            }
            if dest_widget_map["value"].get().strip():
                dest["value"] = float(dest_widget_map["value"].get().strip())
            if dest_widget_map["slippage"].get().strip():
                dest["slippage"] = float(dest_widget_map["slippage"].get().strip())
            destinations_config.append(dest)

        sleeptime = float(self.sleeptime_entry.get())
        start_time = (
            self.start_time_entry.get() if self.start_time_entry.get() else None
        )
        end_time = self.end_time_entry.get() if self.end_time_entry.get() else None

        self.log_message("Starting Trade Copier...")
        if hasattr(self, "trade_copier") and self.trade_copier:
            self.trade_copier.stop()
        try:
            self.trade_copier = TradeCopier(
                source_config,
                destinations_config,
                sleeptime=sleeptime,
                start_time=start_time,
                end_time=end_time,
                custom_logger=logger,
            )

            threading.Thread(target=self.trade_copier.run, daemon=True).start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.log_message("Trade Copier started.")
        except Exception as e:
            messagebox.showerror("Error Starting Copier", str(e))
            self.log_message(f"Error starting copier: {e}")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def stop_copier(self):
        self.log_message("Attempting to stop Trade Copier ...")
        if self.trade_copier and self.trade_copier.running:
            try:
                self.trade_copier.stop()
                time.sleep(5)  # Give some time for the thread to stop
                if not self.trade_copier.running:
                    self.log_message("Trade Copier  stopped successfully.")
            except Exception as e:
                self.log_message(f"Error stopping copier : {e}")
                messagebox.showerror("Error Stopping Copier", str(e))
        else:
            self.log_message("Trade Copier not running or already stopped.")

        self.trade_copier = None  # Reset the copier instance
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def browse_path(self, path_entry_widget):
        filetypes = (("Executable files", "*.exe"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(
            title="Select MetaTrader Terminal Executable", filetypes=filetypes
        )
        if filepath:
            path_entry_widget.delete(0, tk.END)
            path_entry_widget.insert(0, filepath)

    def browse_symbols_file(self, symbols_entry_widget):
        """
        Opens a file dialog to select a .txt file, reads the content,
        and populates the given symbols_entry_widget with a comma-separated list.
        """
        filetypes = (("Text files", "*.txt"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(
            title="Select Symbols File", filetypes=filetypes
        )
        if filepath:
            try:
                with open(filepath, "r") as f:
                    # Read all lines, strip whitespace from each, filter out empty lines
                    lines = [line.strip() for line in f.readlines()]
                    # Join the non-empty lines with a comma
                    symbols_string = "\n".join(filter(None, lines))

                # Update the entry widget
                symbols_entry_widget.delete(0, tk.END)
                symbols_entry_widget.insert(0, symbols_string)
                self.log_message(f"Loaded symbols from {filepath}")
            except Exception as e:
                messagebox.showerror(
                    "Error Reading File", f"Could not read symbols from file: {e}"
                )
                self.log_message(f"Error loading symbols: {e}")


def main():
    """
    Main function to initialize and run the Trade Copier GUI.
    """
    try:
        root = tk.Tk()
        root.iconbitmap(os.path.abspath(ICON_PATH))
        app = TradeCopierApp(root)

        def on_closing():
            if app.trade_copier and app.trade_copier.running:
                app.log_message("Window closed, stopping Trade copier...")
                app.stop_copier()
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        error_details = f"{e}\n\n{traceback.format_exc()}"
        messagebox.showerror("Fatal Error", error_details)
        sys.exit(1)


if __name__ == "__main__":
    main()
