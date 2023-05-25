import pygmsh
from src.models.problemdata import ProblemData
from src.meshes.simplexmesh import SimplexMesh

# ================================================================ #
class Application:
    def __init__(self, ncomp=1, h=None, has_exact_solution=False, random_exactsolution=False, scal_glob={}):
        self.h = h
        self.has_exact_solution = has_exact_solution
        self.random_exactsolution = random_exactsolution
        self.problemdata = ProblemData()
        self.problemdata.ncomp = ncomp
        for k,v in scal_glob.items():
            self.problemdata.params.scal_glob[k] = v
    # def createMesh(self, h=None):
    #     if h is None: h = self.h
    #     return SimplexMesh(mesh=self._createMesh(h))

    def createMesh(self, h=None):
        if h is None: h = self.h
        with pygmsh.geo.Geometry() as geom:
            self.defineGeometry(geom, h)
            mesh = geom.generate_mesh()
        return SimplexMesh(mesh)
