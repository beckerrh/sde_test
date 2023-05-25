import meshio

#----------------------------------------------------------------#
def plotmesh1d(mesh, **kwargs):
    raise NotImplementedError()

#----------------------------------------------------------------#
def plotmesh2d(mesh, **kwargs):
    raise NotImplementedError()

# ----------------------------------------------------------------#
def plotmesh3d(mesh, **kwargs):
    raise NotImplementedError()

#----------------------------------------------------------------#
def plotmesh(mesh, **kwargs):
    if isinstance(mesh, meshio.Mesh):
        celltypes = [key for key, cellblock in mesh.cells]
        assert celltypes==list(mesh.cells_dict.keys())
        if 'tetra' in celltypes:
            args = {'x':mesh.points[:,0], 'y':mesh.points[:,1], 'z':mesh.points[:,2], 'tets':mesh.cells_dict['tetra']}
            plotmesh3d(**args, **kwargs)
        elif 'triangle' in celltypes:
            args = {'x':mesh.points[:,0], 'y':mesh.points[:,1], 'tris': mesh.cells_dict['triangle']}
            plotmesh2d(**args, **kwargs)
        else:
            assert 0
    else:
        try:
            dim = mesh.dimension
            meshdataismesh = True
        except:
            dim = len(mesh) - 3
            meshdataismesh = False
        return dim, meshdataismesh

        if dim == 1:
            plotmesh1d(mesh=mesh, **kwargs)
        elif dim == 2:
            plotmesh2d(mesh=mesh, **kwargs)
        else:
            plotmesh3d(mesh=mesh, **kwargs)
    # if not 'ax' in kwargs or kwargs['ax']==plt: plt.show()
