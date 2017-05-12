import glob
import os

import matplotlib.pyplot as plt
import salem

from .itmix import find_path
from .itmix_cfg import DATA_DIR, ITMIX_ODIR, PLOTS_DIR

pdir = os.path.join(PLOTS_DIR, 'submitted') + '/'
if not os.path.exists(pdir):
    os.mkdir(pdir)

for dgn in glob.glob(os.path.join(ITMIX_ODIR, '*')):
    gname = os.path.basename(dgn)
    print(gname)

    ifile = find_path(os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted'),
                      '02_surface_' + gname + '*.asc')
    ds = salem.EsriITMIX(ifile)
    itmix_topo = ds.get_vardata()

    ifiles = find_path(ITMIX_ODIR, '*' + gname + '*.asc', allow_more=True)
    for ifile in ifiles:
        ds2 = salem.EsriITMIX(ifile)
        oggm_topo = ds2.get_vardata()

        thick = itmix_topo - oggm_topo

        cm = salem.Map(ds.grid)
        cm.set_plot_params(nlevels=256)
        cm.set_cmap(plt.get_cmap('viridis'))
        cm.set_data(thick)
        cm.visualize()

        pname = os.path.basename(ifile).split('.')[0]
        plt.savefig(os.path.join(pdir, pname) + '.png')
        plt.close()