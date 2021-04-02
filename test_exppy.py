#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from exppy import (RandomDesign,
                   LHSDesign,
                   GSDDesign,
                   Experiment,
                   run_experiments)


# spec = (('H_0',   (0.05,   0.5,       'log10', 5)),
#         ('E_0',   (20000., 50000000., 'log10', 5)),
#         ('v_0',   (0.05,   0.45,      'log10', 3)),
#         ('rho_0',  2.4),
#         ('eta_0', 'eta_1'),
#         ('H_1',   '10.-H_0'),
#         ('E_1',   (20000., 5000000.,  'log10', 5)),
#         ('v_1',   (0.05,   0.45,      'log10', 3)),
#         ('rho_1', 1.8),
#         ('eta_1', (20.,    100000.,   'log10', 5)))


spec = (('A',   (1, 10, 'log10',   5)),
        ('B',   (2, 4,  'uniform', 10)),
        ('C',   2.1),
        ('D',   'A+G_1**2'),
        ('E',   (1, 10, 'log2', 5)),
        ('F',   (1, 10, 'log',  5)),
        ('G_1', (1, 100, 'uniform',  5)),
        ('G_2', (1, 100, 'uniform',  5)),
        ('G_3', (1, 100, 'uniform',  5)))


# Define dummy design
class MyDesign(LHSDesign):
    spec = spec
    samples = 20


# Define dummy model
class MyModel:
    def solve(self, d):
        G = d.G
        G_1 = d.G_1
        return {'X': np.random.rand(1), 'Y': G[2]}


class TestDesign(object):
    spec = spec

    def test_random_design(self):
        doe = RandomDesign(self.spec, samples=0.3)
        doe = RandomDesign(self.spec, samples=20)
        d = doe.pick_random_design()
        i, d = doe.pick_random_design(index=True)
        p = doe.pick_random_point()

    def test_lhs_design(self):
        doe = LHSDesign(self.spec, samples=0.3)
        doe = LHSDesign(self.spec, samples=20)
        d = doe.pick_random_design()
        i, d = doe.pick_random_design(index=True)
        p = doe.pick_random_point()

    def test_gsd_design(self):
        doe = GSDDesign(self.spec, samples=0.3)
        doe = GSDDesign(self.spec, samples=20)
        d = doe.pick_random_design()
        i, d = doe.pick_random_design(index=True)
        p = doe.pick_random_point()

    def test_class_spec_and_options(self):
        class MyDesign(LHSDesign):
            spec = self.spec
            samples = 20
        doe = MyDesign()

    def test_experiment(self):
        design = MyDesign()
        model = MyModel()
        exp = Experiment(design, model)
        exp.run()
        X = exp.result.X
        assert isinstance(X, np.ndarray)
        # Below functionality should be in exp.run()
        # Test run previous experiment
        run_experiments(exp.dirname)  # Nothing shoud be done
        aaae = np.testing.assert_array_almost_equal
        aaae(X, np.loadtxt(exp.dirname+'/X.txt')[:, None])
        # Restart experiment
        run_experiments(exp.dirname, restart_from=5)
        aaae(X[:5], np.loadtxt(exp.dirname+'/X.txt')[:5, None])
        assert abs(X[6] - np.loadtxt(exp.dirname+'/X.txt')[6, None]) > 1e-8
        # run in new folder
        run_experiments(exp.dirname+'2', MyDesign, MyModel)
        # Clean
        import shutil
        shutil.rmtree(exp.dirname, ignore_errors=True)
        shutil.rmtree(exp.dirname + '2', ignore_errors=True)


if __name__ == '__main__':
    pytest.main([str(__file__), '-v'])
