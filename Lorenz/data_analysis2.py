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
        self.dirs_selected = []
        frame_button = tk.Frame(root, bg=color)
        button = ttk.Button(frame_button, text="open (dir)", command=partial(self.open, single=False))
        button.pack(side="left", expand=True, fill=tk.X)
        button = ttk.Button(frame_button, text="open (single)", command=partial(self.open, single=True))
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
    def set_dirs(self, fe, destroy=False):
        self.dirs_selected = [fe.fsobjects[f] for f in fe.treeview.get_checked()]
        # print(f"set_dirs: {self.dirs_selected=}")
        if destroy: self.toplevel.destroy()
        self.plot_all()
    def open(self, single=False):
        initialdir = pathlib.Path.home().joinpath('data_dir')
        if single == True:
            dirname = filedialog.askdirectory(initialdir=initialdir)
            self.dirs_selected = [pathlib.Path(dirname)]
            self.plot_all()
        else:
            dirname = filedialog.askdirectory(initialdir=self.initialdir)
            if dirname == '': return

            self.toplevel = tk.Toplevel()
            self.toplevel.title("choose dirs")
            self.toplevel.geometry('400x400')
            self.root.eval(f'tk::PlaceWindow {str(self.toplevel)} center')

            frame = tk.Frame(self.toplevel)
            bframe = tk.Frame(frame)
            cframe = tk.Frame(frame)
            self.file_explorer = tools.file_explorer.FileExplorer(cframe, onlydirs=True, rename = lambda x:x.replace('@','  '))
            self.file_explorer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            cframe.pack(side="top", expand=True, fill=tk.BOTH, padx=5, pady=5)
            button = ttk.Button(bframe, text="done", command=partial(self.set_dirs, fe=self.file_explorer, destroy=True))
            button.pack(side="left", expand=True, fill=tk.BOTH)

            button = ttk.Button(bframe, text="plot", command=partial(self.set_dirs, fe=self.file_explorer, destroy=False))
            button.pack(side="left", expand=True, fill=tk.BOTH)

            bframe.pack(side="top", expand=False, fill=tk.X)
            frame.pack(side="top", expand=True, fill=tk.BOTH)
            ct = self.file_explorer.treeview
            ct.delete(*ct.get_children())
            self.file_explorer.load_tree(pathlib.Path(dirname))
    def plot_all(self):
        if len(self.dirs_selected) == 0:
            messagebox.showwarning("Warning", f"no files slected!")
            return
        dirs = {}
        for dir in self.dirs_selected:
            dirs[str(dir.parent.name)+"@"+dir.name] = dir
        f2 = {k: k.split('@') for k in dirs.keys()}
        inderem = []
        for i in range(len(list(f2.values())[0])):
            if len(set(v[i] for v in f2.values()))==1:
                inderem.append(list(f2.values())[0][i])
        self.inderem = inderem
        for v in f2.values():
            for i in inderem: v.remove(i)
        dirs = {'@'.join(f2[k]):v for k,v in dirs.items()}
        for k,plotter in self.plotters.items():
            fig, canvas = self.canvases[k].figure, self.canvases[k]
            fig.clear()
            try:
                plotter(fig, dirs)
            except:
                messagebox.showwarning("Warning", f"Could not load {dirs=}")
            canvas.draw()
        self.root.focus_force()



# -------------------------------------------------------------------------------- #
from src.lorenz import LorenzTransformed
app = LorenzTransformed()
root = tk.Tk()
root.wm_title("Lorenz")
last2d = partial(app.plot2d, name="ul")
last2d.__name__ = "last2d"
last3d = partial(app.plot3d, name="ul")
last3d.__name__ = "last3d"
dtk = DataTk(root, app, plotters=[app.plot2d, app.plot3d, last2d, last3d, app.histogram1d, app.histogram2d, app.plot_energy])
tk.mainloop()