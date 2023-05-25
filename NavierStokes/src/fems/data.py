#=================================================================#
class BdryData(object):
    """
    Information for boundary conditions
    """
    def __init__(self):
        self.bsaved = {}
        self.Asaved = {}

    def __repr__(self):
        return ", ".join("'{}': {}".format(attr, value) for attr, value in self.__dict__.items())

#=================================================================#
class ConvectionData(object):
    """
    Information for boundary conditions
    """
    def __init__(self, **kwargs):
        self.betacell, self.betart, self.md = kwargs.pop('betacell',None), kwargs.pop('betart',None), kwargs.pop('md',None)
