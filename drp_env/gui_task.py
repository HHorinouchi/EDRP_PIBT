import numpy as np
import matplotlib

_TK_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
    from matplotlib.figure import Figure  # noqa: F401
    matplotlib.use("TkAgg")
    _TK_AVAILABLE = True
except (ImportError, RuntimeError):
    # tkinter が使えない環境（サーバー等）では描画バックエンドを Agg に切り替えて GUI を無効化
    matplotlib.use("Agg")
    tk = None  # type: ignore
    FigureCanvasTkAgg = None  # type: ignore
    Figure = None  # type: ignore

class GUI_tasklist():
    def __init__(self):
        self.headless = not _TK_AVAILABLE
        if self.headless:
            self.root = None
            return
        self.root = tk.Tk()
        self.root.title("Simulation with Task Manager")
        self.init_gui()
        
    def init_gui(self):
        if self.headless:
            return
        # Create the main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for task management
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right frame for task management
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # GUI for task management in left frame
        self.assigned_task_label = ttk.Label(left_frame, text="Tasks Assigned")
        self.assigned_task_label.pack(padx=10, pady=10)

        self.assigned_task_listbox = tk.Listbox(left_frame, height=20, width=30)
        self.assigned_task_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # GUI for task management in right frame (empty column)
        self.task_label = ttk.Label(right_frame, text="Tasks to Assign")
        self.task_label.pack(padx=10, pady=10)

        self.task_listbox = tk.Listbox(right_frame, height=20, width=30)
        self.task_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def update_gui(self):
        """Updates the GUI elements with the current state."""
        if self.headless:
            return
        self.populate_assigned_task_listbox()
        self.root.update_idletasks()

    def populate_assigned_task_listbox(self, agent_num, assigned_tasklist):
        if self.headless:
            return
        # Clear the current content
        self.assigned_task_listbox.delete(0, tk.END)

        # Insert tasks into the assigned_task_listbox
        for i in range(agent_num):
            if len(assigned_tasklist[i])>0:
                task = [assigned_tasklist[i][0], assigned_tasklist[i][1]]
            else:
                task = [-1,-1]

            self.assigned_task_listbox.insert(tk.END, f"agetn{i}: {task[0]} -> {task[1]}")

    def populate_task_listbox(self, current_tasklist, assigned_list):
        if self.headless:
            return
        self.task_listbox.delete(0, tk.END)
        for i in range(len(current_tasklist)):
            if assigned_list[i] == -1:
                self.task_listbox.insert(tk.END, f"{current_tasklist[i][0]} -> {current_tasklist[i][1]}")
            else:
                self.task_listbox.insert(tk.END, f"{current_tasklist[i][0]} -> {current_tasklist[i][1]} : assigned to agent{assigned_list[i]}")

    def show_tasklist(self, agent_num, assigned_tasklist, current_tasklist, assigned_list):
        if self.headless:
            return
        self.populate_assigned_task_listbox(agent_num, assigned_tasklist)
        self.populate_task_listbox(current_tasklist, assigned_list)
        self.root.update_idletasks()