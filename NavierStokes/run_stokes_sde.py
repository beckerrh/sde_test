import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from functools import partial
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
    def raise_popups(self):
        for w in self.root.winfo_children():
            if w.winfo_class() ==  "Toplevel":
                w.tkraise()
    def b_a_w(self):
        if self.app.BaW:
            self.app.BaW = False
            self.button_baw.config(text="black/white")
        else:
            self.app.BaW = True
            self.button_baw.config(text="color")
        self.plot_all()
    def __init__(self, root, app, plotters=None):
        self.root = root
        self.app = app
        if plotters is None: plotters = [plot]
        self.plotters = {p.__name__:p for p in plotters}
        nb = ttk.Notebook(self.root)
        self.nb = nb
        color = tkcolor((160, 160, 160), 0.3)
        self.nbframes, self.canvases, self.figs = {}, {}, {}
        self.dir_selected = ''
        frame_label = tk.Frame(root, bg=color)
        self.text_dir_selected = tk.StringVar()
        label = ttk.Label(frame_label, textvariable=self.text_dir_selected, background='yellow')
        label.pack(side="left", expand=True, fill=tk.X)
        frame_label.pack(fill=tk.X, expand="no", padx=5, pady=5)
        frame_button = tk.Frame(root, bg=color)
        button = ttk.Button(frame_button, text="open (dir)", command=self.open)
        button.pack(side="left", expand=True, fill=tk.X)
        self.app.BaW = False
        button = ttk.Button(frame_button, text="black/white", command=self.b_a_w)
        button.pack(side="left", expand=True, fill=tk.X)
        self.button_baw = button
        frame_button.pack(fill=tk.X, expand="no", padx=5, pady=5)
        for k,plotter in self.plotters.items():
            self.nbframes[k] = tk.Frame(nb, bg=color)
            self.nbframes[k].pack(fill="both", expand="yes", padx=5, pady=5)
            nb.add(self.nbframes[k], text=k)
            # plot notebook
            self.figs[k] = Figure(figsize=(5, 4), dpi=100)
            self.canvases[k] = FigureCanvasTkAgg(self.figs[k], master=self.nbframes[k])  # A tk.DrawingArea.
            canvas = self.canvases[k]
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        nb.pack(fill="both", expand="yes", padx=5, pady=5)
        # menubar
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="Quit", command=self.quit, accelerator="Command-q")
        menu_file.add_command(label="Save single", command=self.save_single)
        menu_file.add_command(label="Save all", command=self.save, accelerator="Command-s")
        menu_file.add_command(label="Raise PupUps", command=self.raise_popups, accelerator="Command-R")
        menubar.add_cascade(label="File", menu=menu_file)
        menu_plot = tk.Menu(menubar, tearoff=0)
        self.plotters_checked = []
        for k,plotter in self.plotters.items():
            item = tk.IntVar()
            item.set(1)
            self.plotters_checked.append(item)
            menu_plot.add_checkbutton(label=f"{k}", variable=item)
        menubar.add_cascade(label="Plot", menu=menu_plot)

        self.root.config(menu=menubar)
        self.initialdir = pathlib.Path.home().joinpath('data_dir')
    def quit(self):
        # self.root.destroy()
        self.root.quit()

    def save_single(self):
        if not hasattr(self, 'inderem'): return
        index = self.nb.index(self.nb.select())
        k = list(self.plotters.keys())[index]
        print(f"{index=} {k=}")
        filename = filedialog.asksaveasfilename(initialdir=self.initialdir, initialfile='_'.join(self.inderem))
        filename += k + ".png"
        self.canvases[k].figure.savefig(pathlib.Path(filename))
    def save(self):
        if not hasattr(self, 'inderem'): return
        # dirname = filedialog.asksaveasfilename(initialdir=self.initialdir, initialfile='_'.join(self.inderem))
        dirname = filedialog.asksaveasfilename(initialdir=self.initialdir)
        dirname += '_'.join(self.inderem)
        # print(f"{dirname=}")
        dir = pathlib.Path(dirname)
        dir.mkdir()
        # print(f"{dir=}")
        for k,plotter in self.plotters.items():
            filename = k + ".png"
            self.canvases[k].figure.savefig(dir/filename)
    def open(self, single=False):
        initialdir = pathlib.Path.home().joinpath('data_dir')
        dirname = filedialog.askdirectory(initialdir=initialdir)
        self.dir_selected = pathlib.Path(dirname)
        self.text_dir_selected.set(str(self.dir_selected))
        self.app.init_from_directory(self.dir_selected)
        self.plot_all()
    def plot_all(self):
        for k,plotter in self.plotters.items():
            fig, canvas = self.canvases[k].figure, self.canvases[k]
            fig.clear()
            plotter(fig)
            canvas.draw()
        self.root.focus_force()



# -------------------------------------------------------------------------------- #
from NavierStokes import stokessde
app = stokessde.RunStokes(noinit=True)
root = tk.Tk()
root.wm_title(app.__class__.__name__)
plot_rates_v = partial(app.plot_rates, var='v')
plot_rates_v.__name__ = "plot_rates_v"
plot_rates_p = partial(app.plot_rates, var='p')
plot_rates_p.__name__ = "plot_rates_p"
dtk = DataTk(root, app, plotters=[plot_rates_v, plot_rates_p, app.plot_errors, app.plot_solution, app.plot_normL2])
dtk.open()
tk.mainloop()