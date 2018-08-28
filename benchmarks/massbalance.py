import os
import shutil
import numpy as np
from oggm.tests.funcs import init_hef, get_test_dir
from oggm import utils, tasks
from oggm.core import massbalance


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


def time_PastMassBalance():

    mb_mod = massbalance.PastMassBalance(gdir, bias=0)
    for yr in years:
        mb_mod.get_annual_mb(heights, year=yr)


def time_ConstantMassBalance():

    mb_mod = massbalance.ConstantMassBalance(gdir, bias=0)
    for yr in years:
        mb_mod.get_annual_mb(heights, year=yr)


def time_RandomMassBalance():

    mb_mod = massbalance.RandomMassBalance(gdir, bias=0)
    for yr in years:
        mb_mod.get_annual_mb(heights, year=yr)


def time_get_ela():

    mb_mod = massbalance.PastMassBalance(gdir, bias=0)
    mb_mod.get_ela(year=years)


time_PastMassBalance.setup = setup
time_PastMassBalance.teardown = teardown
time_ConstantMassBalance.setup = setup
time_ConstantMassBalance.teardown = teardown
time_RandomMassBalance.setup = setup
time_RandomMassBalance.teardown = teardown
time_get_ela.setup = setup
time_get_ela.teardown = teardown
