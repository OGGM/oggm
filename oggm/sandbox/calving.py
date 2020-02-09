from oggm import cfg, utils
from oggm import workflow
from oggm import tasks
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import MassBalanceModel
import numpy as np
import pandas as pd
import netCDF4

def single_flowline_glacier_directory_with_calving(rgi_id, reset=False, prepro_border=10, k_calving=2):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier
    reset : bool
        set to true to delete any pre-existing files. If false (the default),
        the directory won't be re-downloaded if already available locally in
        order to spare time.
    prepro_border : int
        the size of the glacier map: 10, 80, 160, 250

    Returns
    -------
    a GlacierDirectory object
    """

    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    if 'RGI60-' not in rgi_id:
        raise ValueError('OGGM currently expects IDs to start with RGI60-')

    cfg.initialize()
    wd = utils.gettempdir(dirname='pygem-{}-b{}-k'.format(rgi_id,
                                                          prepro_border,
                                                          k_calving),
                          reset=reset)
    cfg.PATHS['working_dir'] = wd
    cfg.PARAMS['use_multiple_flowlines'] = False

    # Check if folder is already processed
    try:
        gdir = utils.GlacierDirectory(rgi_id)
        gdir.read_pickle('model_flowlines')
        # If the above works the directory is already processed, return
        return gdir
    except OSError:
        pass

    # If not ready, we download the preprocessed data for this glacier
    gdirs = workflow.init_glacier_regions([rgi_id],
                                          from_prepro_level=2,
                                          prepro_border=prepro_border)
    if not gdirs[0].is_tidewater:
        raise ValueError('This glacier is not tidewater!')

    # Compute all the stuff
    list_talks = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.catchment_area,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_bedshape,
    ]
    for task in list_talks:
        # The order matters!
        workflow.execute_entity_task(task, gdirs)

    from oggm.core.inversion import find_inversion_calving
    cfg.PARAMS['inversion_calving_k'] = k_calving
    df = find_inversion_calving(gdirs[0])
    print('Calving results:')
    print('k calving inversion:', k_calving)
    for k, v in df.items():
        print(k + ':', v)

    list_talks = [
        tasks.init_present_time_glacier,
    ]
    for task in list_talks:
        # The order matters!
        workflow.execute_entity_task(task, gdirs)

    return gdirs[0]


rid = 'RGI60-01.03622'
gdir = single_flowline_glacier_directory_with_calving(rid, k_calving=2)

diags = gdir.get_diagnostics()
print('Calving results:')
for k in ['calving_front_width', 'calving_flux', 'calving_thick',
          'calving_free_board']:
    print(k + ':', diags[k])

from oggm import graphics
import matplotlib.pyplot as plt
# f, (ax1, ax2) = plt.subplots(1, 2)
# graphics.plot_googlemap(gdir, ax=ax1)
# graphics.plot_inversion(gdir, ax=ax2)
# plt.show()

# Random climate representative for the tstar climate, without bias
# In an ideal world this would imply that the glaciers remain stable,
# but it doesn't have to be so
from oggm.core.massbalance import ConstantMassBalance
from oggm.core.flowline import CalvingModel

mb = ConstantMassBalance(gdir, y0=None, bias=0)
fls = gdir.read_pickle('model_flowlines')

model = CalvingModel(fls, mb_model=mb, do_calving=True, cfl_number=0.01)
df_diag_0 = model.get_diagnostics()

_, ds = model.run_until_and_store(100)
np.testing.assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                           model.flux_gate_m3_since_y0)
df_diag = model.get_diagnostics()

plt.figure()
ds.volume_m3.plot()
plt.show()

