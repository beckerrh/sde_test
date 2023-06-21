import pathlib, sys
import matplotlib.pyplot as plt
SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
from src.tools import timer
from src.models.navierstokes import NavierStokes
from src.models.stokes import Stokes
from src.examples import navier_stokes

# ================================================================c#
def getModel(**kwargs):
    disc_params = kwargs.pop('disc_params', {})
    application = kwargs.pop('application', {})
    model = kwargs.pop('model', 'NavierStokes')
    # bdryplot = kwargs.pop('bdryplot', False)
    # # mesh, data, appname = application.createMesh(), application.problemdata, application.__class__.__name__
    # if bdryplot:
    #     plotmesh.meshWithBoundaries(model.mesh)
    #     plt.show()
    #     return
    # create model
    if model == "Stokes":
        model = Stokes(application=application, disc_params=disc_params)
    else:
        # model = NavierStokes(mesh=mesh, problemdata=data, hdivpenalty=10)
        model = NavierStokes(application=application, disc_params=disc_params, stack_storage=False)
    return model


# ================================================================c#
def static(**kwargs):
    model = getModel(**kwargs)
    newtonmethod = kwargs.pop('newtonmethod', 'newton')
    newtonmaxiter = kwargs.pop('newtonmaxiter', 20)
    appname  = model.application.__class__.__name__
    mesh = model.mesh
    t = timer.Timer("mesh")
    # result, u = model.static(maxiter=30)
    model.linearsolver['disp'] = 0
    model.linearsolver['maxiter'] = 50
    if 'increase_reynolds' in kwargs:
        factor = kwargs['increase_reynolds']
        u = None
        newton_failure = 0
        for i in range(100):
            # try:
                result, u = model.static(u=u, maxiter=newtonmaxiter, method=newtonmethod, rtol=1e-3)
                if not result.newtoninfo.success:
                    newton_failure += 1
                    if newton_failure==3:
                        break
                print(f"{i=} {model.problemdata.params.scal_glob['mu']}\t")
                for k, v in result.data['scalar'].items(): print(k,v,end='\t')
                print()
                model.sol_to_vtu(u=u, suffix=f"_{i:03d}")
                # model.plot(title=f"{model.problemdata.params.scal_glob['mu']}")
                # plt.show()
                model.problemdata.params.scal_glob['mu'] *= factor
                model.new_params()
            # except:
            #     print(f"min viscosity found {model.problemdata.params.scal_glob['mu']}")
            #     break
    else:
        result, u = model.static(maxiter=2*newtonmaxiter, method=newtonmethod, rtol=1e-6)
        print(f"{result}")
        model.plot()
        model.sol_to_vtu()
        plt.show()
# ================================================================c#
def dynamic(**kwargs):
    model = getModel(**kwargs)
    appname  = model.application.__class__.__name__
    stokes = Stokes(application=model.application, stack_storage=False)
    result, u = stokes.solve()
    T = kwargs.pop('T', 200)
    dt = kwargs.pop('dt', 0.52)
    nframes = kwargs.pop('nframes', int(T/2))
    kwargs_dynamic = {'t_span':(0, T), 'nframes':nframes, 'dt':dt, 'theta':0.8, 'output_vtu': False}
    if kwargs.pop('semi_implicit', False):
        kwargs_dynamic['semi_implicit'] = True
    kwargs_dynamic['newton_verbose'] = False

    # result = model.dynamic(u, t_span=(0, T), nframes=nframes, dt=dt, theta=0.8, rtolnewton=1e-3, output_vtu=True)
    result = model.dynamic(u, **kwargs_dynamic)
    print(f"{model.timer=}")
    print(f"{model.newmatrix=}")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"{appname}")
    gs = fig.add_gridspec(2, 3)
    nhalf = (nframes - 1) // 2
    for i in range(3):
        model.plot(fig=fig, gs=gs[i], iter = i*nhalf, title=f't={result.time[i*nhalf]}')
    pp = model.get_postprocs_dynamic()
    ax = fig.add_subplot(gs[1, :])
    for k,v in pp['postproc'].items():
        ax.plot(pp['time'], v, label=k)
    ax.legend()
    ax.grid()
    plt.show()
    # def initfct(ax, u):
    #     ax.set_aspect(aspect=1)
    # anim = animdata.AnimData(mesh, v, plotfct=model.plot_v, initfct=initfct)
    # plt.show()

#================================================================#
if __name__ == '__main__':
    test = 'st_2d_dyn'
    # test = 'dc_2d_stat'
    # test = 'dc_3d_stat'
    # test = 'dc_2d_dyn'
    if test == 'st_2d_stat':
        app = navier_stokes.SchaeferTurek2d(h=0.2)
        static(application=app)
        # app = navier_stokes.Poiseuille2d(h=0.2, mu=0.1)
    elif test == 'st_2d_dyn':
        app = navier_stokes.SchaeferTurek2d(h=0.3, mu=0.002, errordrag=False)
        dynamic(application=app, T=50, dt=0.25)
        # dynamic(application=app, T=50, dt=0.25, semi_implicit=True)
        # app = navier_stokes.Poiseuille2d(h=0.2, mu=0.1)
    elif test == 'st_3d_stat':
        app = navier_stokes.SchaeferTurek3d(h=0.5)
        # app = navier_stokes.SchaeferTurek3d(h=0.5, mu=0.01)
        # app = navier_stokes.Poiseuille3d(h=0.5, mu=0.01)
        app = navier_stokes.BackwardFacingStep3d(h=0.1, mu=0.001)
    elif test == 'st_3d_dyn':
        app = navier_stokes.SchaeferTurek3d(h=0.5, mu=0.01)
        # app = navier_stokes.Poiseuille3d(h=0.5, mu=0.01)
        # app = navier_stokes.BackwardFacingStep3d(h=0.1, mu=0.001)
        dynamic(application=app, T=50, dt=0.25)
    elif test == 'dc_2d_stat':
        app = navier_stokes.DrivenCavity2d(h=0.1)
        static(application=app)
    elif test == 'dc_2d_dyn':
        app = navier_stokes.DrivenCavity2d(h=0.1, mu=1e-4)
        dynamic(application=app, T=50, dt=0.25)
    elif test == 'dc_3d_stat':
        app = navier_stokes.DrivenCavity3d(h=0.2)
        static(application=app)
