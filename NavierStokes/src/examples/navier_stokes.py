import numpy as np
import pygmsh
from src.models.problemdata import ProblemData
from src.meshes.simplexmesh import SimplexMesh
import src.models.application

class Application(src.models.application.Application):
    def __init__(self, ncomp=2, h=None, mu=0.1):
        super().__init__(ncomp=ncomp, h=h, scal_glob={'mu':mu})
# ================================================================ #
class Poiseuille2d(Application):
    def __init__(self, mu=0.01, h=0.5):
        super().__init__(mu=mu, h=h)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1002, 1000, 1003])
        self.problemdata.bdrycond.set("Neumann", [1001])
        self.problemdata.bdrycond.set("Navier", [])
        self.problemdata.bdrycond.set("Pressure", [])
        self.problemdata.bdrycond.fct[1003] = [lambda x, y, z: 4 * y * (1 - y), lambda x, y, z: 0]
        # parameters
        self.problemdata.params.scal_glob["navier"] = 1.01
        # TODO pass ncomp with mesh ?!
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
# ================================================================ #
class Poiseuille3d(Application):
    def __init__(self, mu=0.01, h=0.5):
        super().__init__(mu=mu, h=h, ncomp=3)
        data = self.problemdata
        # boundary conditions
        data.bdrycond.set("Dirichlet", [100, 103])
        data.bdrycond.set("Neumann", [101])
        data.bdrycond.fct[103] = [lambda x, y, z: 16 * y * (1 - y) * z * (1 - z), lambda x, y, z: 0, lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top, p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(vol, label="10")
# ================================================================c#
def drivenCavity2d(h=0.1, mu=0.01):
    with pygmsh.geo.Geometry() as geom:
        ms = [h*v for v in [1.,1.,0.2,0.2]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
        geom.add_physical(p.surface, label="100")
        geom.add_physical(p.lines[2], label="1002")
        geom.add_physical([p.lines[0], p.lines[1], p.lines[3]], label="1000")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1002])
    data.bdrycond.fct[1002] = [lambda x, y, z: 1, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = mu
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data
# ================================================================c#
def drivenCavity3d(h=0.1, mu=0.01):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical(top, label="102")
        geom.add_physical([p.surface, lat[0], lat[1], lat[2], lat[3]], label="100")
        geom.add_physical(vol, label="10")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [100, 102])
    data.bdrycond.fct[102] = [lambda x, y, z: 1, lambda x, y, z: 0, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = mu
    data.ncomp = 3
    return SimplexMesh(mesh=mesh), data
# ================================================================ #
class BackwardFacingStep2d(Application):
    def __init__(self, mu=0.02, h=0.2):
        super().__init__(mu=mu, h=h)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1000, 1002])
        # self.problemdata.bdrycond.set("Pressure", [1004])
        self.problemdata.bdrycond.set("Neumann", [1004])
        self.problemdata.bdrycond.fct[1000] = [lambda x, y, z: y * (1 - y), lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        hs = 6*[h]
        hs[2] *= 0.2
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=hs)
        #  np.insert(np.array(X), 2, 0, axis=1): fills zeros for z-coord
        geom.add_physical(p.surface, label="100")
        dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        geom.add_physical(dirlines, "1002")
        geom.add_physical(p.lines[0], "1000")
        geom.add_physical(p.lines[4], "1004")
# ================================================================ #
class BackwardFacingStep3d(Application):
    def __init__(self, h=0.2, mu=0.02):
        super().__init__(mu=mu, h=h, ncomp=3)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [100, 102])
        self.problemdata.bdrycond.set("Neumann", [104])
        self.problemdata.bdrycond.fct[102] = [lambda x, y, z: y*(1-y)*z*(1-z), lambda x, y, z: 0, lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        hs = 6*[h]
        hs[2] *= 0.1
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=hs)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        dirf = [lat[i] for i in range(1,6) if i!=4 ]
        dirf.extend([p.surface, top])
        geom.add_physical(dirf, label="100")
        geom.add_physical(lat[0], label="102")
        geom.add_physical(lat[4], label="104")
        # for i in range(len(lat)):
        #     geom.add_physical(lat[i], label=f"{101+i}")
        geom.add_physical(vol, label="10")
        # geom.add_physical(p.surface, label="100")
        # dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        # geom.add_physical(dirlines, "1002")
        # geom.add_physical(p.lines[0], "1000")
        # geom.add_physical(p.lines[4], "1004")
# ================================================================ #
class SchaeferTurek2d(Application):
    def __init__(self, hcircle=None, mu=0.01, h=0.5, errordrag=True):
        super().__init__(mu=mu, h=h)
        self.hcircle, self.errordrag = hcircle, errordrag
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1002, 1000, 1003, 3000])
        self.problemdata.bdrycond.set("Neumann", [1001])
        self.problemdata.bdrycond.fct[1003] = [lambda x, y, z: 0.3 * y * (4.1 - y) / 2.05 ** 2, lambda x, y, z: 0]
        self.problemdata.params.scal_glob["mu"] = mu
        self.problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=3000)
        self.problemdata.postproc.plot = ['drag', 'lift1', 'lift2']
    def changepostproc(self, info):
        bdrynflux = info.pop('bdrynflux_3000')
        # print(f"changepostproc: {bdrynflux=}")
        info['drag'] = -50 * bdrynflux[0]
        info['lift'] = -50 * bdrynflux[1]
        if self.errordrag:
            info['err_drag'] = 5.57953523384 + 50 * bdrynflux[0]
            info['err_lift'] = 0.010618937712 + 50 * bdrynflux[1]
    def defineGeometry(self, geom, h):
        if self.hcircle is None: hcircle = 0.2 * h
        circle = geom.add_circle(x0=[2,2], radius=0.5, mesh_size=hcircle, num_sections=10, make_surface=False)
        geom.add_physical(circle.curve_loop.curves, label="3000")
        p = geom.add_rectangle(xmin=0, xmax=11, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
# ================================================================ #
class SchaeferTurek3d(Application):
    def __init__(self, hcircle=None, mu=0.01, h=0.5, errordrag=True):
        super().__init__(mu=mu, h=h, ncomp=3)
        self.hcircle, self.errordrag = hcircle, errordrag
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [100, 103, 300])
        self.problemdata.bdrycond.set("Neumann", [101])
        self.problemdata.bdrycond.fct[103] = [lambda x, y, z: 0.45 * y * (4.1 - y) * z * (4.1 - z) / 2.05 ** 4, lambda x, y, z: 0,
                                  lambda x, y, z: 0]
        self.problemdata.params.scal_glob["mu"] = mu
        self.problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=300)
    def changepostproc(self, info):
        bdrynflux = info.pop('bdrynflux_300')
        scale = 50/4.1
        info['drag'] = -scale * bdrynflux[0]
        info['lift1'] = -scale * bdrynflux[1]
        info['lift2'] = -scale * bdrynflux[2]
        if self.errordrag:
            info['err_drag'] = 6 + scale * bdrynflux[0]
            info['err_lift1'] = 0.01 + scale * bdrynflux[1]
            info['err_lift2'] = 0.0175 + scale * bdrynflux[2]
    def defineGeometry(self, geom, h):
        if self.hcircle is None: hcircle = 0.3 * h
        circle = geom.add_circle(x0=[5,2], radius=0.5, mesh_size=hcircle, num_sections=8, make_surface=False)
        p = geom.add_rectangle(xmin=0, xmax=15, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        axis = [0, 0, 4.1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top,p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(lat[4:], label="300")
        geom.add_physical(vol, label="10")
