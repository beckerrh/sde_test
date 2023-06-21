import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib


#----------------------------------------------------------------#
class AnimData:
    def __init__(self, fig, axs, data, pltfct, blit=True):
        self.fif, self.data, self.axs, self.pltfct = fig, data, axs, pltfct
        self.nframes = len(data)
        self.animation = animation.FuncAnimation(fig, self, frames=self.nframes, blit=blit)
        self.paused = False
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
    def __call__(self, i):
        # actors=[]
        # self.title.set_text(f"Iter {i + 1}/{self.nframes}")
        return self.pltfct(self.axs, self.data[i])

        for ia,ax in enumerate(self.axs):
            ax.cla()
            ax.set_title(f"Iter {i + 1}/{self.nframes}")
            act = self.pltfcts[ia](ax, self.data[i][ia])
            # actors.extend(list(act))
        # print(f"{actors=}")
        return self.axs
        return [*self.axs, self.title]