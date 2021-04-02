# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 21:30:46 2014

@author: mwojc
"""
import numpy as np
import collections
import random
import string
import traceback
import os
import pickle
import shutil
import sys
import warnings
import ast


def is_valid_variable_name(name):
    if not isinstance(name, str):
        return False
    if '.' in name:
        return False
    try:
        ast.parse('{} = None'.format(name))
        return True
    except Exception:
        return False


class VarDict(collections.abc.MutableMapping):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        if not self.is_valid_key(key):
            raise KeyError("Key not allowed: '%s'" % key)
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return 'VarDict({})'.format(self.__dict__)

    def is_valid_key(self, key):
        return is_valid_variable_name(key) and (key not in dir(dict))


class IntDict(collections.abc.MutableMapping):
    def __init__(self, N):
        self.N = N
        self.__dict = dict()

    def __setitem__(self, key, value):
        if not self.is_valid_key(key):
            raise KeyError("Key must be an integer from 0 to %i" % self.N)
        self.__dict[key] = value

    def __getitem__(self, key):
        return self.__dict[key]

    def __delitem__(self, key):
        del self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __str__(self):
        return str(self.__dict)

    #def __repr__(self):
    #    return '{}, VarDict({})'.format(super().__repr__(), self.__dict__)

    def is_valid_key(self, key):
        return isinstance(key, int) and key < self.N
    
    def tolist(self):
        return [self.get(i, None) for i in range(self.N)]

    # def toarray(self):
    #     lst = self.tolist()
    #     nones = 



def _get_by_name_or_idx(obj, key, maxkey):
    if isinstance(key, str):
        return obj.__dict__[key]
    elif isinstance(key, int):
        if abs(key) >= maxkey:
            raise KeyError
        d = VarDict()
        for k, v in obj.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v[key]
            else:
                d[k] = v
        return d
    else:
        raise KeyError


def _savetxt(obj, dirname='.'):
    # Create dir
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # Array formatting
    fmt = {'S': '%s', 'd': '%.18e', 'l': '%i'}
    # Dump data
    for k, v in obj.__dict__.items():
        if not k.startswith('_'):
            name = dirname + '/' + str(k) + '.txt'
            try:
                warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
                varray = np.asarray(v)
                warnings.simplefilter('default', np.VisibleDeprecationWarning)
                varray_fmt = fmt[varray.dtype.char]
                np.savetxt(name, varray, fmt=varray_fmt)
            except (ValueError, IndexError, TypeError, KeyError):
                with open(name, 'w') as f:
                    f.write(str(v))


def _random_string(length):
    return ''.join(random.choice(string.ascii_letters)
                   for m in range(length))


class Design(object):
    name = None
    spec = None

    def __init__(self, spec=None, **kwargs):
        """
        Example of variable spec:

        spec = (('H_0',   (0.05,   0.5,       'log10', 5)),
                ('E_0',   (20000., 50000000., 'log10', 5)),
                ('v_0',   (0.05,   0.45,      'log10', 3)),
                ('rho_0',  2.4),
                ('eta_0', 'eta_1'),
                ('H_1',   '10.-H_0'),
                ('E_1',   (20000., 5000000.,  'log10', 5)),
                ('v_1',   (0.05,   0.45,      'log10', 3)),
                ('rho_1', 1.8),
                ('eta_1', (20.,    100000.,   'log10', 5)))
        """
        spec = self.spec if spec is None else spec
        if spec is None:
            raise ValueError("No spec")
        else:
            self.spec = spec
        self.names = tuple([v[0] for v in spec])
        self.dnames = tuple([v[0] for v in spec
                             if isinstance(v[1], (tuple, list))])
        self.dbounds = tuple([v[1][:2] for v in spec
                              if isinstance(v[1], (tuple, list))])
        self.dscale = tuple([v[1][2] for v in spec
                             if isinstance(v[1], (tuple, list))])
        self.dnums = tuple([v[1][3] for v in spec
                            if isinstance(v[1], (tuple, list))])
        self.didx = tuple([i for i, v in enumerate(spec)
                           if isinstance(v[1], (tuple, list))])
        self.cnames = tuple([v[0] for v in spec
                             if not isinstance(v[1], (tuple, list))])
        self.cexpr = tuple([str(v[1]) for v in spec
                            if not isinstance(v[1], (tuple, list))])
        self.cidx = tuple([i for i, v in enumerate(spec)
                           if not isinstance(v[1], (tuple, list))])
        self.nvars = len(self.names)
        self.ndvars = len(self.dnames)
        self.ncvars = len(self.cnames)
        self.doe(**kwargs)

    def _scale(self, x, fstr):
        if fstr == 'log10':
            return np.log10(x)
        if fstr == 'log2':
            return np.log2(x)
        if fstr == 'log':
            return np.log(x)
        return x

    def _rescale(self, x, fstr):
        if fstr == 'log10':
            return 10**x
        if fstr == 'log2':
            return 2**x
        if fstr == 'log':
            return np.exp(x)
        return x

    def _doe2design(self, doe):
        # Create design array
        design = np.empty((len(doe), self.nvars))
        design[:, self.didx] = doe
        # Evaluate constant and dependent variables
        kvars = dict(zip(self.names, design.T))
        for i in range(self.ncvars):
            cname = self.cnames[i]
            cexpr = self.cexpr[i]
            kvars[cname][:] = eval(cexpr, kvars)
            cidx = self.cidx[i]
            design.T[cidx] = kvars[cname]
        return design

    def _combine_vars(self):
        vnames = ()
        vidx = ()
        kvars = dict(zip(self.names, self.design.T))
        nvars = collections.OrderedDict()
        for i, name in enumerate(self.names):
            k = name.rfind('_')
            nam, num = name[:k], name[k+1:]
            if num.isdigit():
                nvars.setdefault(nam, []).append((i, int(num)))
        for k, v in nvars.items():
            idx, nums = np.array(v).T
            if np.allclose(np.sort(nums)-nums.min(), np.arange(len(nums))):
                i = idx[nums.argsort()]
                vnames += (k,)
                vidx += (tuple(i),)
                kvars[k] = self.design[:, i]
        return kvars, vnames, vidx

    def _create_design(self, doe):
        # Evaluate design
        self.design = self._doe2design(doe)
        self.ndesign = len(self.design)
        # Combine vars into vectors and update self.__dict__
        kvars, self.vnames, self.vidx = self._combine_vars()
        self.__dict__.update(kvars)
        # Create also scaled values
        self.dvars = self.design[:, self.didx]
        self.cvars = self.design[:, self.cidx]
        dvars_scaled = self.dvars.copy()
        for i in range(self.ndvars):
            fstr = self.dscale[i]
            dvars_scaled[:, i] = self._scale(self.dvars[:, i], fstr)
        self.dvars_scaled = dvars_scaled

    def get(self, key):  # ordering is lost
        return _get_by_name_or_idx(self, key, len(self.design))

    def totxt(self, dirname='.'):
        _savetxt(self, dirname)

    def pick_random_design(self, index=False):
        idx = np.random.randint(0, self.ndesign)
        return self.get(idx) if not index else idx, self.get(idx)

    def pick_random_point(self):
        design = RandomDesign(self.spec, samples=1)
        return design.get(0)


class RandomDesign(Design):
    name = 'RANDOM'
    samples = None

    def doe(self, **kwargs):
        samples = kwargs.get('samples', self.samples)
        if samples is None:
            samples = 0.05
        if isinstance(samples, float):
            assert samples < 1.
            samples = int(samples * np.prod(self.dnums)) + 1
        doe = np.zeros((samples, self.ndvars))
        for i in range(self.ndvars):
            b0, b1 = self.dbounds[i]
            fstr = self.dscale[i]
            b0 = self._scale(b0, fstr)
            b1 = self._scale(b1, fstr)
            doe.T[i] = np.random.uniform(b0, b1, samples)
            doe.T[i] = self._rescale(doe.T[i], fstr)
        self._create_design(doe)


class LHSDesign(Design):
    name = 'LHS'
    samples = None
    criterion = 'corr'
    iterations = 10

    def doe(self, **kwargs):
        samples = kwargs.get('samples', self.samples)
        criterion = kwargs.get('criterion', self.criterion)
        iterations = kwargs.get('iterations', self.iterations)
        if samples is None:
            samples = 0.05
        if isinstance(samples, float):
            assert samples < 1.
            samples = int(samples * np.prod(self.dnums)) + 1
        from pyDOE2 import lhs
        doe = lhs(self.ndvars, samples, criterion, iterations)
        for i in range(self.ndvars):
            b0, b1 = self.dbounds[i]
            fstr = self.dscale[i]
            b0 = self._scale(b0, fstr)
            b1 = self._scale(b1, fstr)
            doe.T[i] = self._rescale(b0 + doe.T[i]*(b1-b0), fstr)
        self._create_design(doe)


class GSDDesign(Design):
    name = 'GSD'
    reduction = 5

    def doe(self, **kwargs):
        reduction = kwargs.get('reduction', self.reduction)
        from pyDOE2 import gsd
        doeidx = gsd(self.dnums, reduction)
        doe = np.empty(doeidx.shape)
        for i in range(self.ndvars):
            b0, b1 = self.dbounds[i]
            fstr = self.dscale[i]
            b0 = self._scale(b0, fstr)
            b1 = self._scale(b1, fstr)
            n = self.dnums[i]
            dvals = np.linspace(b0, b1, n)
            idx = doeidx.T[i]
            doe.T[i] = self._rescale(dvals[idx], fstr)
        self._create_design(doe)


class Result(object):
    def __init__(self, N):
        self._N = N

    def _create_empty(self, k, v):
        z = np.zeros_like(v)
        self.__dict__[k] = np.asarray([z]*self._N)

    def set(self, i, r):  # r must be dict like
        d = self.__dict__
        for k, v in r.items():
            v = np.asarray(v)
            if k not in d:
                self._create_empty(k, v)
            # handle variable size strings
            if v.dtype.char == 'S' and d[k].itemsize < v.itemsize:
                oldv = d[k]
                self._create_empty(k, v)
                d[k][:] = oldv
            self.__dict__[k][i] = v

    def get(self, key):
        return _get_by_name_or_idx(self, key, self._N)

    def totxt(self, dirname='.'):
        _savetxt(self, dirname)


class Model(object):
    def solve(self, d):
        raise AttributeError("Solve method not defined in model")


class Experiment(object):
    status_codes = {-2: 'Exception in model',
                    -1: 'Not yet calculated',
                     0: 'Sucessfull calculation'}

    def __init__(self, design, model, dirname=None):
        self.design = design
        self.model = model
        if hasattr(model, 'status_codes'):
            self.status_codes.update(model.status_codes)
        self.N = len(self.design.design)
        self.result = Result(self.N)
        self.current_design = 0
        self.finished = False
        self.status = np.zeros(self.N, dtype=int) - 1
        if dirname is None:
            dirname = _random_string(8)
        self.dirname = dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.dfname = self.dirname+'/'+self.dirname+'.exp'
        self.rfname = self.dirname+'/report.txt'

    @property
    def success(self):
        return self.status >= 0

    @property
    def N_success(self):
        return self.success.sum()

    @property
    def N_failed(self):
        return (~self.success).sum()

    def get(self, key, success=None):
        if isinstance(key, int):
            raise NotImplementedError
        val = None
        try:
            val = self.design.get(key)
        except KeyError:
            val = self.result.get(key)
        if isinstance(val, np.ndarray) and len(val) == self.N:
            if success is True:
                val = val[self.success]
            if success is False:
                val = val[~self.success]
        return val

    def runi(self, i, allres=False):
        d = self.design.get(i)
        m = self.model
        r = {}
        try:
            res = m.solve(d)
            if allres:
                return res
            else:
                r.update(res[0] if not isinstance(res, dict) else res)
        except Exception:
            r['status'] = -2
            r['message'] = traceback.format_exc()
        if 'status' not in r:
            r['status'] = 0
        return VarDict(r)

    def run(self):
        self.report('\n\nRunning experiments from %i to %i\n'
                    % (self.current_design, self.N))
        for i in range(self.current_design, self.N):
            self.report('\nExperiment *%i*:\n' % i)
            self.current_design = i
            r = self.runi(i)
            status = r.pop('status')
            message = r.pop('message') if 'message' in r else ''
            if status >= 0:  # assign results
                self.result.set(i, r)
            self.status[i] = status  # update status
            if status >= 0:  # no error
                self.dump(backup=True)  # dump with backup
            self.report('Finished with status: \n')
            self.report('% -4i: %s\n' % (status, self.status_codes[status]))
            if message != '':
                self.report(message)
        self.finished = True

        self.dump(backup=True)  # just to record finished value
        self.totxt()

    def dump(self, fname=None, backup=True):
        fname = self.dfname if fname is None else fname
        # First create backup
        if backup and os.path.isfile(fname):
            shutil.copy2(fname, fname+'.bak')
        # Then dump
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        # And remove backup
        if backup and os.path.isfile(fname+'.bak'):
            os.remove(fname+'.bak')

    def report(self, s, fname=None):
        fname = self.rfname if fname is None else fname
        with open(fname, 'a') as f:
            f.write(s)

    def _format_status_codes(self):
        codes = self.status_codes.copy()
        keys = np.sort(list(codes.keys()))
        s = ''
        for k in keys:
            s += '% -4i' % k + ': ' + '%s\n' % codes[k]
        return s

    def totxt(self, dirname=''):
        if dirname == '':
            dirname = self.dirname
        self.design.totxt(dirname)
        self.result.totxt(dirname)
        np.savetxt(dirname+'/status.txt', self.status, fmt='%i')
        np.savetxt(dirname+'/success.txt', self.status, fmt='%i')
        with open(dirname+'/status_codes.txt', 'w') as f:
            f.write(self._format_status_codes())
        with open(dirname+'/finished.txt', 'w') as f:
            f.write(str(self.finished))


def run_experiments(dirname, dcls=None, mcls=None, restart_from=None):
    expname = '%s/%s.exp' % (dirname, dirname)

    if os.path.isfile(expname):
        exp = pickle.load(open(expname, 'rb'))
        if restart_from is not None:
            exp.finished = False  # to be sure
            exp.current_design = restart_from-1
        if not exp.finished:
            exp.current_design += 1  # to start from first unfinished
            exp.report('\n\nRestarting from design %i' % exp.current_design)
            exp.run()
        else:
            print ("Finished! Nothing to do!")
    else:
        if dcls is not None and mcls is not None:
            design = dcls()
            model = mcls()
            exp = Experiment(design, model, dirname=dirname)
            exp.run()
        else:
            raise ValueError("Both, design and model class have to be given!")


def load_experiment(expname):
    dirname = os.path.dirname(os.path.abspath(expname))
    sys.path.append(dirname+'/..')  # BUG: exp should be saved in main dir
    with open(expname, 'rb') as f:
        exp = pickle.load(f)
    return exp


if __name__ == "__main__":
    import pytest
    pytest.main(['test_exppy.py', '-v'])
