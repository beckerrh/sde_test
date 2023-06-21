# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016
@author: becker
"""
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from src.models import problemdata
from src.tools.latexwriter import LatexWriter
import src.tools.timer


#=================================================================#

class Results():
    def __init__(self, names, paramname, parameters, infos):
        self.names = names
        self.paramname = paramname 
        self.parameters = parameters 
        self.infos = infos
        self.errors={}
        for k, v in infos.items():
            if k[:3]=='err': self.errors[k] = v

class CompareMethods(object):
    """
    Run several times a list of methods (typically for comparison of different discretizations on a sequence of meshes)
    possible parameters:
      latex
      vtk
      plot
      plotpostprocs
      verb: in [0,5]
    """
    def __init__(self, **kwargs):
        self.only_iter_and_timer = kwargs.pop("only_iter_and_timer", False)
        self.dirname = os.getcwd() + os.sep +"Results"
        if kwargs.pop("clean",True):
            try: shutil.rmtree(self.dirname)
            except: pass
        if not os.path.isdir(self.dirname): os.mkdir(self.dirname)
        self.verbose = kwargs.pop("verbose", 1)
        self.latex = kwargs.pop("latex", True)
        self.plotsolution = kwargs.pop("plotsolution", False)
        self.plotpostprocs = kwargs.pop("plotpostprocs", False)
        if self.verbose == 0: self.latex = False
        self.paramname = kwargs.pop("paramname", "ncells")
        self.plot = kwargs.pop("plot", False)
        self.application = kwargs.pop("application", None)
        self.geom = kwargs.pop("geom", None)
        self.mesh = kwargs.pop("mesh", None)
        self.postproc = kwargs.pop("postproc", None)
        self.h = kwargs.pop("h", None)
        self.paramsdict = kwargs.pop("paramsdict")
        if self.paramname == "ncells":
            if 'h' in kwargs:
                self.params = kwargs.pop("h")
                self.gmshrefine = False
            elif not 'uniformrefine' in kwargs:
                h1 = kwargs.pop("h1", 1)
                niter = kwargs.pop("niter", 3)
                # if niter is None: raise KeyError("please give 'niter' ({self.paramname=}")
                hred = kwargs.pop("hred", 0.5)
                self.params = [h1*hred**i for i in range(niter)]
                self.gmshrefine = False
            else:
                # raise NotImplementedError(f"gmeshrefine not working")
                # ne marche pas à cause de pygmsh !!!
                mesh = self.application.createMesh()
                self.gmshrefine = True
                niter = kwargs.pop("niter", None)
                if niter is None: raise KeyError("please give 'niter' ({self.paramname=}")
                self.params = [mesh.ncells*mesh.dimension**i for i in range(niter)]
        else:
            # self.params = kwargs.pop("params", None)
            self.params = self.paramsdict[self.paramname]
            self.gmshrefine = False
        if 'methods' in kwargs:
            self.methods = kwargs.pop("methods")
            if len(kwargs.keys()):
                raise ValueError(f"*** unused arguments {kwargs=}")
        else:
            requiredargs = ['model', 'modelargs']
            for requiredarg in requiredargs:
                if not requiredarg in kwargs:
                    raise ValueError("need 'model' (class) and 'modelargs' (dict) and  'paramsdict' (dict)")
            model, modelargs = kwargs.pop("model"), kwargs.pop("modelargs")
            if len(kwargs.keys()):
                raise ValueError(f"*** unused arguments {kwargs=}")
            self._definemethods(model, modelargs)

    def _definemethods(self, model, modelargs):
        # if not 'problemdata' in modelargs:
        #     raise KeyError(f"'problemdata' should be set in 'modelargs'")
        paramsdict = self.paramsdict
        if self.paramname in paramsdict: paramsdict.pop(self.paramname)
        for pname,params in paramsdict.items():
            if isinstance(params, str): paramsdict[pname] = [params]
        def dicttensorproduct(paramsdicts):
            import itertools
            paramslist = [[(name, param) for param in params] for name, params in paramsdicts.items()]
            return [{p[0]: p[1] for p in params} for params in itertools.product(*paramslist)]
        paramslist = dicttensorproduct(paramsdict)
        self.methods = {}
        import copy
        sep = '@'
        for i,p in enumerate(paramslist):
            name = ''
            modelargs2 = copy.deepcopy(modelargs)
            problemdataparamchange = problemdata.Params()
            for pname, param in p.items():
                # print(f"{pname=} {param=} {len(paramsdict[pname])=}")
                if isinstance(param,list) and len(param)==2:
                    if hasattr(self,'names2names'):
                        raise ValueError(f"paramsdict should all be in the form {{paramname:[name,param]}} or {{paramname:param}} got {pname=} {param=}")
                    name += param[0] + sep
                    modelargs2[pname] = param[1]
                else:
                    ps = pname.split('@')
                    if len(ps)>1:
                        name += f"{ps[0]}={str(param)}" + sep
                        exec(f"problemdataparamchange.{ps[1]}['{ps[0]}']={param}")
                        # exec(f"modelargs2['problemdata'].params.{ps[1]}['{ps[0]}']={param}")
                    else:
                        modelargs2[pname] = param
                        name += str(param) + sep
                    if isinstance(param,dict) and len(paramsdict[pname]) > 1:
                        if not hasattr(self, 'names2names'): self.names2names={}
            name = name[:-1]
            modelargs2['application'] = copy.deepcopy(self.application)
            model_of_name = model(**modelargs2)
            model_of_name.problemdata.params.update(problemdataparamchange)
            if hasattr(self, 'names2names'):
                self.names2names[f"{i}"] = name
                key = f"{i}"
            else:
                key = name
            self.methods[key] = model(**modelargs2)
            self.methods[key].problemdata.params.update(problemdataparamchange)

    def compare(self, **kwargs):
        if (self.gmshrefine or self.paramname != "ncells") and self.mesh is None:
            mesh = self.application.createMesh()
        if self.plotsolution:
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(10, 8))
            outer = gridspec.GridSpec(1, len(self.params)*len(self.methods), wspace=0.6, hspace=0.3)
            plotcount = 0
        parameters = []
        for iter, param in enumerate(self.params):
            if self.verbose: print(f"{self.__class__.__name__} {iter:2d} {self.paramname=} {param=}")
            if self.paramname == "ncells":
                if self.gmshrefine:
                    mesh = src.meshes.pygmshext.gmshRefine(mesh)
                else:
                    # raise ValueError(f"{mesh=}")
                    mesh = self.application.createMesh(param)
                print(f"{self.__class__.__name__} {mesh=} {param=}")
                parameters.append(mesh.ncells)
            else:
                parameters.append(param)
            for name, method in self.methods.items():
                if self.verbose: print(f"\t{self.__class__.__name__} {name=}")
                method.setMesh(mesh)
                self.dim = mesh.dimension
                if self.paramname != "ncells": 
                    method.paramname = param
                    # method.setParameter(self.paramname, param)
                result,u = method.solve()
                # print(f"{result=}")
                if self.plotsolution:
                    suptitle = "{}={}".format(self.paramname, parameters[-1])
                    method.plot(u=u, fig=fig, gs=outer[plotcount])
                    plotcount += 1
                    # plt.show()
                resdict = result.info.copy()
                if self.postproc: self.postproc(result.data['scalar'])
                resdict.update(result.data['scalar'])
                self.fillInfo(iter, name, resdict, len(self.params))
        if self.plotsolution:
            import os
            plt.savefig(os.path.join(self.dirname,"toto.png")) 
            plt.show()
        if self.plotpostprocs:
            self.plotPostprocs(self.methods.keys(), self.paramname, parameters, self.infos)
        if self.latex:
            self.generateLatex(self.methods.keys(), self.paramname, parameters, self.infos)
        return  Results(self.methods.keys(), self.paramname, parameters, self.infos)
    def fillInfo(self, iter, name, info, n):
        if not hasattr(self, 'infos'):
            # first time - we have to generate some data
            self.infos = {}
            for key2, info2 in info.items():
                self.infos[key2] = {}
                if isinstance(info2, dict):
                    for key3, info3 in info2.items():
                        self.infos[key2][key3] = {}
                        for name2 in self.methods.keys():
                            self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
                elif isinstance(info2, src.tools.timer.Timer):
                    for key3, info3 in info2.data.items():
                        self.infos[key2][key3] = {}
                        for name2 in self.methods.keys():
                            self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
                else:
                    for name2 in self.methods.keys():
                        self.infos[key2][name2] = np.zeros(shape=(n), dtype=type(info2))
        for key2, info2 in info.items():
            if isinstance(info2, dict):
                for key3, info3 in info2.items():
                    self.infos[key2][key3][name][iter] = np.sum(info3)
            elif isinstance(info2, src.tools.timer.Timer):
                for key3, info3 in info2.data.items():
                    self.infos[key2][key3][name][iter] = np.sum(info3)
            else:
                self.infos[key2][name][iter] = np.sum(info2)
    def generateLatex(self, names, paramname, parameters, infos, title=None):
        if title is None:
            title = self.application.__class__.__name__
            # title = f"mesh({mesh})\\\\"
            # for name, method in self.methods.items():
            #     title += f"{name}\\\\"
            # title = title[:-2]
        # print("title = ", title)
        latexwriter = LatexWriter(dirname=self.dirname, title=title, author=self.__class__.__name__)
        for key, val in infos.items():
            kwargs = {'n': parameters, 'nname': paramname}
            keysplit = key.split('_')
            if key == 'iter':
                newdict={}
                for key2, val2 in val.items():
                    for name in names:
                        keyname = "{}-{}".format(key2, name)
                        newdict[keyname] = val2[name]
                kwargs['name'] = '{}'.format(key)
                kwargs['values'] = newdict
                latexwriter.append(**kwargs)
            elif key == 'timer':
                sumdict = {name:np.zeros(len(parameters)) for name in names}
                for name in names:
                    newdict={}
                    for key2, val2 in val.items():
                        sumdict[name] += val2[name]
                        newdict[key2] = val2[name]
                    if not self.only_iter_and_timer:
                        latexwriter.append(**kwargs, name = f"{key}-{name}", values=newdict, percentage=True)
                latexwriter.append(**kwargs, name=key, values=sumdict)
            else:
                iserr = len(keysplit) >= 2 and keysplit[0][:3] == 'err'
                # print(f"{iserr=} {keysplit=} {keysplit[0][:3]=}")
                kwargs['redrate'] = iserr and (paramname=="ncells")
                kwargs['diffandredrate'] = not kwargs['redrate'] and (paramname=="ncells")
                kwargs['dim'] = self.dim
                kwargs['name'] = '{}'.format(key)
                kwargs['values'] = val
                latexwriter.append(**kwargs)

        if hasattr(self, 'names2names'): latexwriter.write(names2names=self.names2names)
        else: latexwriter.write()
        latexwriter.compile()
    def computeOrder(self, ncells, values, dim):
        fnd = float(ncells[-1]) / float(ncells[0])
        order = -dim * np.log(values[-1] / values[0]) / np.log(fnd)
        return np.power(ncells, -order / dim), np.round(order,2)
    def plotPostprocs(self, names, paramname, parameters, infos):
        nmethods = len(names)
        self.reds = np.outer(np.linspace(0.2,0.8,nmethods),[0,1,1])
        self.reds[:,0] = 1.0
        self.greens = np.outer(np.linspace(0.2,0.8,nmethods),[1,0,1])
        self.greens[:,1] = 1.0
        self.blues = np.outer(np.linspace(0.2,0.8,nmethods),[1,1,0])
        self.blues[:,2] = 1.0
        singleplots = ['timer', 'iter']
        nplotsc = len(infos.keys())
        nplotsr = 0
        for key, val in infos.items():
            if key in singleplots: number=1
            else: number=len(val.keys())
            nplotsr = max(nplotsr, number)
        fig, axs = plt.subplots(nplotsr, nplotsc, figsize=(nplotsc * 3, nplotsr * 3), squeeze=False)
        cc = 0
        for key, val in infos.items():
            cr = 0
            for key2, val2 in val.items():
                for name in names:
                    if key == "error":
                        axs[cr,cc].loglog(parameters, val2[name], '-x', label="{}_{}".format(key2, name))
                        if self.paramname == "ncells":
                            orders, order = self.computeOrder(parameters, val2[name], self.dim)
                            axs[cr, cc].loglog(parameters, orders, '-', label="order {}".format(order))
                    # else:
                    #     axs[cr, cc].plot(parameters, val2[name], '-x', label="{}_{}".format(key2, name))
                axs[cr, cc].legend()
                if key not in singleplots:
                    axs[cr, cc].set_title("{} {}".format(key, key2))
                    cr += 1
            if key in singleplots:
                axs[cr, cc].set_title("{}".format(key))
                cr += 1
            cc += 1
        plt.tight_layout()
        plt.show()
# ------------------------------------- #
if __name__ == '__main__':
    print("so far no test")