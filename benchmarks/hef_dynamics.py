import os
import shutil
import numpy as np
from oggm.tests.funcs import init_hef, get_test_dir
from oggm import utils, tasks
from oggm.core import massbalance, flowline


testdir = os.path.join(get_test_dir(), 'benchmarks')
utils.mkdir(testdir, reset=True)
heights = np.linspace(2200, 3600, 120)
years = np.arange(151) + 1850


def teardown():
    if os.path.exists(testdir):
        shutil.rmtree(testdir)


def setup():
    global gdir
    gdir = init_hef(border=80)
    teardown()
    gdir = tasks.copy_to_basedir(gdir, base_dir=testdir, setup='all')
    flowline.init_present_time_glacier(gdir)


def time_hef_run_until():

    mb_mod = massbalance.RandomMassBalance(gdir, bias=0, seed=0)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.)
    model.run_until(200)


def time_hef_run_until_in_steps():

    mb_mod = massbalance.RandomMassBalance(gdir, bias=0, seed=0)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.)
    for yr in np.linspace(0, 200, 400):
        model.run_until(yr)


def time_hef_run_until_and_store():

    mb_mod = massbalance.RandomMassBalance(gdir, bias=0, seed=0)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.)
    model.run_until_and_store(200)


def time_hef_run_until_and_store_with_nc():

    mb_mod = massbalance.RandomMassBalance(gdir, bias=0, seed=0)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.)
    model.run_until_and_store(200, run_path=os.path.join(testdir, 'run.nc'),
                              diag_path=os.path.join(testdir, 'diag.nc'))


time_hef_run_until.setup = setup
time_hef_run_until.teardown = teardown

time_hef_run_until_in_steps.setup = setup
time_hef_run_until_in_steps.teardown = teardown

time_hef_run_until_and_store.setup = setup
time_hef_run_until_and_store.teardown = teardown

time_hef_run_until_and_store_with_nc.setup = setup
time_hef_run_until_and_store_with_nc.teardown = teardown
