import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from functools import partial, update_wrapper
import pathlib, sys

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
import tools.file_explorer

def plot(ax, files):
    ax.clear()
    for key,file in files.items():
        u = np.load(file)
        xy, z = 0.5*(u[:,0]+u[:,1]), u[:,2]
        ax.plot(xy, z, label=key)
    ax.legend()

def tkcolor(rgb, alpha=1):
    return "#" + ''.join(f"{int(255 - alpha * (255 - c)):02x}" for c in rgb)

# -------------------------------------------------------------------------------- #
class DataTk():
    def __init__(self, root, plotters=None):
        self.root = root
        if plotters is None: plotters = [plot]
        self.plotters = {p.__name__:p for p in plotters}
        nb = ttk.Notebook(self.root)
        self.nb = nb
        color = tkcolor((160, 160, 160), 0.3)
        self.nbframes, self.canvases, self.figs = {}, {}, {}

        self.files_selected = {}
        for k,plotter in self.plotters.items():
            self.files_selected[k] = []
            self.nbframes[k] = tk.Frame(nb, bg=color)
            frame_button = tk.Frame(self.nbframes[k], bg=color)
            frame_plot = tk.Frame(self.nbframes[k], bg=color)
            button = ttk.Button(frame_button, text="open (dir)", command=partial(self.open, k=k, dir=True))
            button.pack(side="left", expand=True, fill=tk.X)
            button = ttk.Button(frame_button, text="open (files)", command=partial(self.open, k=k, dir=False))
            button.pack(side="left", expand=True, fill=tk.X)
            # button = ttk.Button(frame_button, text="plot", command=partial(self.plot_selected, k=k))
            # button.pack(side="left", expand=True, fill=tk.X)
            frame_button.pack(fill="both", expand="yes", padx=5, pady=5)

            frame_button.pack(fill="both", expand="yes", padx=5, pady=5)
            frame_plot.pack(fill="both", expand="yes", padx=5, pady=5)

            self.nbframes[k].pack(fill="both", expand="yes", padx=5, pady=5)
            nb.add(self.nbframes[k], text=k)
            # plot notebook
            self.figs[k] = Figure(figsize=(5, 4), dpi=100)
            self.canvases[k] = FigureCanvasTkAgg(self.figs[k], master=frame_plot)  # A tk.DrawingArea.
            canvas = self.canvases[k]
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # menubar
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar, tearoff=0)
        # menu_file.add_command(label="Open", command=self.open, accelerator="Command-O")
        # self.root.bind_all("<Command-o>", lambda event: self.open())
        self.root.createcommand("::tk::mac::Quit", self.quit)
        menu_file.add_command(label="Quit", command=self.quit)
        menu_file.add_command(label="Save", command=self.save, accelerator="Command-S")
        self.root.createcommand("::tk::mac::Save", self.save)
        menubar.add_cascade(label="File", menu=menu_file)

        menu_plot = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plot", menu=menu_plot)
        # for p in plotters:
        #     command = partial(self.plot_selected, p)
        #     menu_plot.add_command(label=p.__name__, command=command, accelerator="Command-P")
        # self.root.bind_all("<Command-p>", lambda event: command)
        self.root.config(menu=menubar)

        nb.pack(fill="both", expand="yes", padx=5, pady=5)

    def quit(self):
        self.root.destroy()
    def save(self):
        k = self.active_plot
        # print(f"{k=}")
        filename = k
        for item in self.file_explorer.treeview.get_checked():
            filename += f"_{item}"
        filename = filedialog.asksaveasfilename(defaultextension=".png", initialdir=self.initialdir, initialfile=filename)
        self.canvases[k].figure.savefig(filename)
    def set_files(self, k, fe):
        self.files_selected[k] = [fe.fsobjects[f] for f in fe.treeview.get_checked()]
        # print(f"{self.files_selected[k]=}")
        self.toplevel.destroy()
        self.plot_selected(k)
    def open(self, k, dir=False):
        initialdir = pathlib.Path.home().joinpath('data_dir')
        if dir == False:
            filenames = filedialog.askopenfilenames(initialdir=initialdir)
            self.files_selected[k] = [pathlib.Path(f) for f in filenames]
            self.plot_selected(k)
        else:
            dirname = filedialog.askdirectory(initialdir=initialdir)
            if dirname == '': return
            self.toplevel = tk.Toplevel()
            self.toplevel.title(k)
            frame = tk.Frame(self.toplevel)
            bframe = tk.Frame(frame)
            cframe = tk.Frame(frame)
            self.file_explorer = tools.file_explorer.FileExplorer(cframe)
            self.file_explorer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            cframe.pack(side="top", expand=True, fill=tk.X)
            button = ttk.Button(bframe, text="done", command=partial(self.set_files, k=k, fe=self.file_explorer))
            button.pack(side="left", expand=True, fill=tk.X)
            bframe.pack(side="top", expand=True, fill=tk.X)
            frame.pack(side="top", expand=True, fill=tk.BOTH)
            # button.wait_variable(self.done_pushed)
            ct = self.file_explorer.treeview
            ct.delete(*ct.get_children())
            self.initialdir = pathlib.Path(dirname)
            self.file_explorer.load_tree(self.initialdir)
    def plot_selected(self, k):
        if len(self.files_selected[k]) == 0:
            messagebox.showwarning("Warning", f"no files slected!")
            return
        files = {}
        for file in self.files_selected[k]:
            files[str(file.parent.name)+"@"+file.name[:-4]] = file
        f2 = {k: k.split('@') for k in files.keys()}
        inderem = []
        for i in range(len(list(f2.values())[0])):
            if len(set(v[i] for v in f2.values()))==1:
                inderem.append(list(f2.values())[0][i])
        for v in f2.values():
            for i in inderem: v.remove(i)
        files = {'@'.join(f2[k]):v for k,v in files.items()}
        fig, canvas = self.canvases[k].figure, self.canvases[k]
        fig.clear()
        self.plotters[k](fig, files)
        canvas.draw()
        self.nb.select(self.nbframes[k])
        self.active_plot = k


# -------------------------------------------------------------------------------- #
from cython_test.lorenz import LorenzTransformed
app = LorenzTransformed()
root = tk.Tk()
root.wm_title("Lorenz")
# hist1d = update_wrapper(partial(app.plot_histogram, dim=1), app.plot_histogram)
# hist2d = update_wrapper(partial(app.plot_histogram, dim=2), app.plot_histogram)
hist1d = partial(app.plot_histogram, dim=1)
hist1d.__name__ = "hist1d"
hist2d = partial(app.plot_histogram, dim=2)
hist2d.__name__ = "hist2d"
dtk = DataTk(root, plotters=[app.plot2d, app.plot3d, hist1d, hist2d])
tk.mainloop()