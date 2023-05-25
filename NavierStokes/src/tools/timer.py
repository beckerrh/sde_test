import time

#=================================================================#
class Timer():
    def __del__(self):
        if self.verbose>1: print(self)
    def __init__(self, name='', verbose=0):
        self.name = name
        self.verbose = verbose
        self.tlast = time.time()
        self.nameslast = 'none'
        self.data = {}
        self.counter = 0
    def __repr__(self):
        tall = sum(self.data.values())
        repr = f"Timer({self.name:}) total = {tall:8.2e}\n"
        for name, t in self.data.items():
            repr += f"\t{name:12s}: {100*t/tall:5.1f}%  ({t:8.2e})\n"
        return repr[:-1]
    # def items(self): return self.data.items()
    def add(self, name=None):
        if name is None: name = str(self.counter); self.counter += 1
        if name not in self.data: self.data[name] = 0
        t = time.time()
        self.data[name] += t - self.tlast
        if self.verbose: print(f"{name=} {self.nameslast=} {(t - self.tlast)=}")
        self.tlast = t
        self.nameslast = name
    def reset(self, name):
        self.data[name] = 0
    def reset_all(self):
        for name in self.data: self.data[name] = 0
        self.tlast = time.time()
