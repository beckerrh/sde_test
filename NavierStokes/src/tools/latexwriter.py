# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import os, platform
import numpy as np


#=================================================================#
class TableData(object):
    """
    n : first axis
    values : per method
    """
    def __repr__(self):
        s =  f"{self.nname=} {self.nformat=}\n{self.n=}"
        for k in self.values:
            s += f"\n{self.values[k]=}\n{self.valformat[k]=}"
        return s
    def _getformat(self,v):
        assert len(v)
        isint = (isinstance(v,list) and isinstance(v[0], (int, np.integer))) or (isinstance(v,np.ndarray) and np.issubdtype(v.dtype, np.integer))
        isfloat = (isinstance(v,list) and isinstance(v[0], (float, float))) or (isinstance(v,np.ndarray) and np.issubdtype(v[0], float))
        if isint:
            format = "{:15d}"
        elif isfloat:
            v = np.asarray(v)
            if np.all(np.abs(v) < 100) and np.abs(v).min()>0.01:
                format = "{:10.2f}"
            else:
                format = "{:10.2e}"
        elif isinstance(v[0], str):
            format = "{:15s}"
        else:
            raise ValueError(f"cannot find instance of {v=} {type(v[0])=} {isinstance(v,np.ndarray)=} {np.issubdtype(v.dtype,np.float)=}")
        return format
    def __init__(self, **kwargs):
        values = kwargs.pop('values')
        if not isinstance(values, dict):
            raise ValueError("values is not a dictionary (values=%s)" %values)
        self.values = dict((str(k), v) for k, v in values.items())
        self.n = kwargs.pop('n')
        for k,v in values.items():
            if len(self.n) != len(v):
                raise ValueError(f"wrong lengths: n({len(self.n)}) value({len(v)})\n{values=}")
        self.nname = kwargs.pop('nname')
        if 'nformat' in kwargs:
            self.nformat = "{{:{}}}".format(kwargs.pop('nformat'))
        else:
            self.nformat = self._getformat(self.n)
        self.valformat = {}
        if 'valformat' in kwargs:
            valformat = kwargs.pop('valformat')
            if isinstance(valformat,str):
                for k in values.keys():
                    self.valformat[k] = "{{:{}}}".format(valformat)
            else:
                assert len(valformat)== len(self.values)
                for k in values.keys():
                    self.valformat[k] = "{{:{}}}".format(valformat[k])
        else:
             for k,v in self.values.items():
                    self.valformat[k] = self._getformat(v)

    def computePercentage(self):
        self.values = dict((k+"(\%)", v) for k, v in self.values.items())
        self.values['sum'] = np.zeros(len(self.n))
        for i in range(len(self.n)):
            sum = 0
            for key, value in self.values.items():
                sum += self.values[key][i]
            self.values['sum'][i] = sum
            for key, value in self.values.items():
                if key=='sum': continue
                self.values[key][i] *= 100/sum
        for key in self.values.keys():
            self.valformat[key] = "{:8.2f}"
        self.valformat['sum'] = "{:10.2e}"
    def computeDiffs(self):
        n, values, keys = self.n, self.values, list(self.values.keys())
        for key in keys:
            key2 = key + '-d'
            valorder = np.zeros(len(n))
            for i in range(1,len(n)):
                valorder[i] = abs(values[key][i]-values[key][i-1])
            values[key2] = valorder
            self.valformat[key2] = "{:10.2e}"
    def computeReductionRate(self, dim, diff=False):
        n, values, keys = self.n, self.values, list(self.values.keys())
        if not isinstance(n[0],(int,float)): raise ValueError("n must be int or float")
        fi = 1+int(diff)
        for key in keys:
            if diff:
                if key[-2:] != "-d": continue
            key2 = key + '-o'
            valorder = np.zeros(len(n))
            for i in range(fi,len(n)):
                if not values[key][i-1]:
                    p = -1
                    continue
                fnd = float(n[i])/float(n[i-1])
                vnd = values[key][i]/values[key][i-1]
                if abs(vnd)>1e-10:
                    p = -dim* np.log(np.abs(vnd)) / np.log(np.abs(fnd))
                else:
                    p=-1
                valorder[i] = p
            values[key2] = valorder
            self.valformat[key2] = "{:8.2f}"

#=================================================================#
class LatexWriter(object):
    # def __init__(self, dirname="Resultslatextest", filename=None):
    def __init__(self, **kwargs):
        self.author = kwargs.pop("author".replace('_',r'\_'), self.__class__.__name__)
        self.title = kwargs.pop("title".replace('_',r'\_'), "No title given")
        self.dirname = kwargs.pop("dirname", "Resultslatextest")
        filename = kwargs.pop("filename", self.title.replace(' ',r'\_'))
        if filename[-4:] != '.tex': filename += '.tex'
        self.dirname += os.sep + "tex"
        if not os.path.isdir(self.dirname): os.makedirs(self.dirname)
        self.latexfilename = os.path.join(self.dirname, filename)
        # raise ValueError(f"{self.latexfilename=}")
        self.sep = '%' + 30*'='+'\n'
        self.data = {}
        self.countdata = 0
    def __del__(self):
        try:
            self.latexfile.close()
        except:
            pass
    def append(self, **kwargs):
        if 'name' in kwargs: name = kwargs.pop('name')
        else: name = 'table_{:d}'.format(self.countdata+1)
        # print(f"append {name=} {kwargs.values()=}")
        # raise ValueError(f"{name=}")
        self.countdata += 1
        tabledata = TableData(**kwargs)
        if 'diffandredrate' in kwargs and kwargs.pop('diffandredrate'):
            tabledata.computeDiffs()
            tabledata.computeReductionRate(kwargs.pop('dim'), diff=True)
        if 'redrate' in kwargs and kwargs.pop('redrate'):
            tabledata.computeReductionRate(kwargs.pop('dim'))
        if 'percentage' in kwargs and kwargs.pop('percentage'):
            tabledata.computePercentage()
        self.data[name] = tabledata
    def write(self, sort=False, names2names=None):
        self.latexfile = open(self.latexfilename, "w")
        if len(self.data) < 4: self.namestype='short'
        # if len(list(self.data.keys())[0]) > 4: self.namestype='list'
        self.writePreamble()
        if sort:
            for key,tabledata in sorted(self.data.items()):
                self.writeTable(name=key, tabledata=tabledata)
        else:
            for key, tabledata in self.data.items():
                # print(f"{key=} {tabledata=}")
                self.writeTable(name=key, tabledata=tabledata)
        if names2names:
            texta = '\\begin{table}[!htbp]\n\\begin{center}\n\\begin{tabular}{'
            texta += 'r|r' + '}\n'
            self.latexfile.write(texta)
            for k,v in names2names.items():
                vr = v.replace('_', r'\_').replace('{','').replace('}','')
                self.latexfile.write(f"{k}&{vr}\\\\\n")
            texte = f"\n\\end{{tabular}}\n\\caption{{{'name'}}}"
            texte += f"\n\\end{{center}}\n\\label{{tab:{'name'}}}\n\\end{{table}}\n"
            texte += f"%\n%{30*'-'}\n%\n"
            self.latexfile.write(texte)
        self.writePostamble()
        self.latexfile.close()
    def writeTable(self, name, tabledata, sort=False):
        n, nname, nformat, values, valformat = tabledata.n, tabledata.nname, tabledata.nformat, tabledata.values, tabledata.valformat
        nname = nname.replace('_', r'\_')
        name = name.replace('_', '')
        keys_to_write = sorted(values.keys()) if sort else list(values.keys())
        # print(f"{name=} {tabledata.nname=} {keys_to_write=}")
        size = len(keys_to_write)
        if size==0: return
        texta ='\\begin{table}[!htbp]\n\\begin{center}\n\\begin{tabular}{'
        texta += 'r|' + size*'|r' + '}\n'
        self.latexfile.write(texta)
        if max([len(keys_to_write[i]) for i in range(size)])>4: rotate=True
        else: rotate=False
        # print(f"{self.rotate=} {ks=} {max([len(k) for k in kwargs['values'].keys()])=}")
        if rotate:
            # itemformated = "\sw{%s} &" %nname
            itemformated = f"\sw{{{nname}}} &"
            for i in range(size-1):
                # itemformated += "\sw{%s} &" %keys_to_write[i].replace('_','\_')
                k = keys_to_write[i].replace('_', r'\_')
                itemformated += f"\sw{{{k}}} &"
            # itemformated += "\sw{%s}\\\\\\hline\hline\n" %keys_to_write[size-1].replace('_','\_')
            k = keys_to_write[size-1].replace('_', '\_')
            itemformated += f"\sw{{{k}}}\\\\\\hline\hline\n"
        else:
            itemformated = f"{nname:15} "
            for i in range(size):
                k = keys_to_write[i].replace('_', r'\_')
                itemformated += f" & {k:15} "
            itemformated += "\\\\\\hline\hline\n"
        self.latexfile.write(itemformated)
        for texline in range(len(n)):
            nt = n[texline]
            if isinstance(nt,str): nt=nt.replace('_',r'\_')
            itemformated = nformat.format(nt)
            for i in range(size):
                key = keys_to_write[i]
                try:
                    itemformated += '&'+valformat[key].format(values[key][texline])
                except:
                    raise ValueError(f"{key=} {values[key]=} {valformat[key]=}")
            itemformated += "\\\\\\hline\n"
            self.latexfile.write(itemformated)
        # texte = '\\end{tabular}\n\\caption{%s}' %(name)
        texte = f"\\end{{tabular}}\n\\caption{{{name}}}"
        texte += f"\n\\end{{center}}\n\\label{{tab:{name}}}\n\\end{{table}}\n"
        texte += f"%\n%{30*'-'}\n%\n"
        self.latexfile.write(texte)
    def writePreamble(self, name="none"):
        texta = '\\documentclass[11pt]{article}\n\\usepackage[margin=3mm, a4paper]{geometry}\n'
        texta += '\\usepackage{times,graphicx,rotating,subfig}\n'
        texta += "\\newcommand{\sw}[1]{\\begin{sideways} #1 \\end{sideways}}\n"
        texta += f"\\author{{{self.author}}}\n"
        texta += f"\\title{{{self.title}}}\n"
        texta += self.sep + '\\begin{document}\n' + self.sep + '\n'
        texta += "\\maketitle\n"
        texta += f"%\n%{30*'-'}\n%\n"
        self.latexfile.write(texta)
    def writePostamble(self):
        texte = '\n' + self.sep + '\\end{document}\n' + self.sep
        self.latexfile.write(texte)
        self.latexfile.close()
    def compile(self):
        import subprocess
        os.chdir(self.dirname)
        filename = os.path.basename(self.latexfilename)
        command = "pdflatex " + filename
        try:
            result = subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result: raise ValueError("*** command pdflatex not found")
        except:
            raise ValueError("*** no pdflatex found")
        if platform.system() == "Linux":
            command = "xdg-open "
        elif platform.system() == "Darwin":
            command = "open "
        elif platform.system() == "Windows":
            command = "explorer "
        command += filename.replace('.tex', '.pdf')
        try: 
            subprocess.call(command, shell=True)
        except: 
            pass


# ------------------------------------- #
if __name__ == '__main__':
    n = [i**2 for i in range(1, 10)]
    values={}
    values['u'] = np.power(n,-2.) + 0.01*np.random.rand((len(n)))
    values['v'] = np.power(n,-3.) + 0.01*np.random.rand((len(n)))
    latexwriter = LatexWriter()
    latexwriter.append(n=n, nname='n', values=values, diffandredrate=True, dim=2)
    values2={}
    values2[1] = [1,2,3]
    values2[2] = [4,5,6]
    latexwriter.append(n=['a_a','b','c'], nname='letter', values=values2, percentage=True)
    values3={}
    values3['1'] = np.linspace(1,3,5)
    latexwriter.append(n=np.arange(5), nname= 'toto', values=values3)
    latexwriter.write()
    latexwriter.compile()
