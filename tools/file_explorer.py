"""
    Simple File System Explorer with Tk
"""

import tkinter as tk
from tkinter import ttk
import pathlib
from ttkwidgets import CheckboxTreeview


class FileExplorer(ttk.Frame):

    def __init__(self, root, load_dir=None, onlydirs=False, rename=None):
        super().__init__(root)
        self.rename = rename
        if self.rename is None: self.rename = lambda x:x
        self.onlydirs = onlydirs
        self.fsobjects: dict[str, pathlib.Path] = {}
        self.treeview = CheckboxTreeview(self, show="tree")
        if load_dir is not None: self.load_tree(load_dir)

        self.treeview.tag_bind("fstag", "<<TreeviewOpen>>", self.item_opened)
        self.treeview.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def safe_iterdir(self, path: pathlib.Path):
        """
        Like `Path.iterdir()`, but do not raise on permission errors.
        """
        try:
            return tuple(path.iterdir())
        except PermissionError:
            print("You don't have permission to read", path)
            return ()

    def insert_item(self, name: str, path: pathlib.Path, parent: str = ""):
        """
        Insert a file or folder into the treeview and return the item ID.
        """
        # iid = self.treeview.insert(parent, tk.END, text=name, tags=("fstag",),
        #     image=self.get_icon(path))
        if self.onlydirs and path.is_file(): return
        # print(f"{parent=} {self.rename(parent)=}")
        iid = self.treeview.insert(parent, tk.END, text=self.rename(name), tags=("fstag",))
        self.fsobjects[iid] = path
        return iid
    
    def load_tree(self, path: pathlib.Path, parent: str = ""):
        """
        Load the contents of `path` into the treeview. 
        """
        for fsobj in sorted(self.safe_iterdir(path)):
            fullpath = path / fsobj
            child = self.insert_item(fsobj.name, fullpath, parent)
            # Preload the content of each directory within `path`.
            # This is necessary to make the folder item expandable.
            if fullpath.is_dir():
                for sub_fsobj in self.safe_iterdir(fullpath):
                    self.insert_item(sub_fsobj.name, fullpath / sub_fsobj, child)
    
    def load_subitems(self, iid: str):
        """
        Load the content of each folder inside the specified item
        into the treeview.
        """
        for child_iid in self.treeview.get_children(iid):
            if self.fsobjects[child_iid].is_dir():
                self.load_tree(self.fsobjects[child_iid],parent=child_iid)
    
    def item_opened(self, _event: tk.Event):
        """
        Handler invoked when a folder item is expanded.
        """
        # Get the expanded item.
        iid = self.treeview.selection()[0]
        # If it is a folder, loads its content.
        self.load_subitems(iid)

#---------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    fe = FileExplorer(root, load_dir=pathlib.Path.home(), onlydirs=True)
    # fe.load_tree(pathlib.Path.home())
    root.mainloop()
