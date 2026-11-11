"""Command line arguments to the oggm_prepro_addons command

Type `$ oggm_prepro_addons -h` for help

The add-ons are optional gridded data (velocity, thickness, elevation change
products...) for the glacier directories. They are not part of the
processing levels: they are made once from the L1 directories, since they
only depend on the glacier grid, and users add them to any level from L1 to
L4 with `workflow.apply_addons`, which also bins them onto the elevation band
flowlines.
"""

# Standard libraries
import os
import sys
import argparse
import time
import logging
import json
import importlib
from pathlib import Path

# External modules
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow
from oggm.cli.prepro_levels import apply_rgi_fixes, _level_dir
from oggm.exceptions import InvalidParamsError
from oggm.utils import entity_task

log = logging.getLogger(__name__)

# The add-on products: the shop task adding them to gridded_data, the
# variables it writes, the task compiling their statistics, where the data
# comes from, and whether binning preserves their totals (True by default,
# for thicknesses and elevation changes, but meaningless for velocities)
ADDON_PRODUCTS = {
    'consensus': dict(
        task='oggm.shop.bedtopo:add_consensus_thickness',
        variables=['consensus_ice_thickness'],
        statistics='oggm.shop.bedtopo:compile_consensus_statistics',
        source='Farinotti et al. (2019), doi:10.1038/s41561-019-0300-3'),
    'itslive': dict(
        task='oggm.shop.its_live:itslive_velocity_to_gdir',
        variables=['itslive_v', 'itslive_vx', 'itslive_vy'],
        statistics='oggm.shop.its_live:compile_itslive_statistics',
        source='ITS_LIVE, https://its-live.jpl.nasa.gov',
        preserve_totals=False),
    'millan_thickness': dict(
        task='oggm.shop.millan22:millan_thickness_to_gdir',
        variables=['millan_ice_thickness'],
        statistics='oggm.shop.millan22:compile_millan_statistics',
        source='Millan et al. (2022), doi:10.1038/s41561-021-00885-z'),
    'millan_velocity': dict(
        task='oggm.shop.millan22:millan_velocity_to_gdir',
        variables=['millan_v', 'millan_vx', 'millan_vy'],
        statistics='oggm.shop.millan22:compile_millan_statistics',
        source='Millan et al. (2022), doi:10.1038/s41561-021-00885-z',
        preserve_totals=False),
    'hugonnet': dict(
        task='oggm.shop.hugonnet_maps:hugonnet_to_gdir',
        variables=['hugonnet_dhdt'],
        statistics='oggm.shop.hugonnet_maps:compile_hugonnet_statistics',
        source='Hugonnet et al. (2021), doi:10.1038/s41586-021-03436-z'),
    'bedmachine': dict(
        task='oggm.shop.bedmachine:bedmachine_to_gdir',
        variables=['bedmachine_ice_thickness'],
        statistics='oggm.shop.bedmachine:compile_bedmachine_statistics',
        source='BedMachine Greenland and Antarctica, Morlighem et al.'),
}


def _import(path):
    """Import a function from its `module:function` path."""
    module, name = path.split(':')
    return getattr(importlib.import_module(module), name)


@entity_task(log, name_kwarg='label')
def _write_product_addon(gdir, label=None, variables=None, output_dir=None,
                         source=None, version=None, preserve_totals=True):
    """Write the variables of a product found in gridded_data as an add-on.

    A glacier without any of the variables (e.g. outside the product's
    coverage) gets no add-on.
    """
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        found = [v for v in variables if v in nc.variables]
    if not found:
        log.info('(%s) no %s data: no add-on.', gdir.rgi_id, label)
        return
    utils.write_addon(gdir, label=label, variables=found,
                      output_dir=output_dir, source=source, version=version,
                      preserve_totals=preserve_totals)


def run_prepro_addons(rgi_version=None, rgi_reg=None, border=None,
                      products=None, output_folder='', working_dir='',
                      start_base_url=None, start_from_dir=None, version=None,
                      is_test=False, test_ids=None, rgi_file=None,
                      disable_mp=False, params_file=None,
                      override_params=None, logging_level='WORKFLOW',
                      continue_on_error=True):
    """Generate the add-on trees from the L1 glacier directories.

    Each product is written to its own tree,
    ``{output_folder}/{product}/RGI{v}/b_{border:03d}/``, laid out as the
    level trees, with its statistics in the ``summary`` folder. Point
    ``workflow.apply_addons`` (``base_urls``) to ``{output_folder}/{product}``.

    Parameters
    ----------
    rgi_version : str
        the RGI version to use (defaults to cfg.PARAMS)
    rgi_reg : str
        the RGI region to process
    border : int
        the number of pixels at the maps border
    products : list of str
        the products to make, keys of ``ADDON_PRODUCTS``
    output_folder : str
        path to the output folder (where to put the add-on trees)
    working_dir : str
        path to the OGGM working directory
    start_base_url : str
        the base url of the tree holding the L1 directories
    start_from_dir : str
        like start_base_url, but for a tree which is on disk
    version : str
        the version of the add-ons, recorded in their manifests
    is_test : bool
        to test on a couple of glaciers only (see `test_ids`)
    test_ids : list
        if is_test: list of ids to process
    rgi_file : str or geopandas.GeoDataFrame, optional
        an RGI file to use instead of the default RGI region file
    disable_mp : bool
        disable multiprocessing
    params_file : str
        path to the OGGM parameter file (to override defaults)
    override_params : dict
        a dict of parameters to override
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    continue_on_error : bool
        set to False to stop at the first error (for testing)
    """

    products = utils.tolist(products) if products else []
    unknown = [p for p in products if p not in ADDON_PRODUCTS]
    if not products or unknown:
        raise InvalidParamsError(f'products should be some of '
                                 f'{list(ADDON_PRODUCTS)}, got {products}.')
    if start_base_url is None and start_from_dir is None:
        raise InvalidParamsError('The add-ons are made from the L1 '
                                 'directories: set start_base_url or '
                                 'start_from_dir.')

    start = time.time()

    override_params = dict(override_params or {})
    override_params['use_multiprocessing'] = not disable_mp
    override_params['border'] = border
    override_params['continue_on_error'] = continue_on_error
    utils.mkdir(working_dir)
    override_params['working_dir'] = working_dir
    cfg.initialize(file=params_file, params=override_params,
                   logging_level=logging_level)

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']

    if rgi_file is None:
        rgidf = gpd.read_file(utils.get_rgi_region_file(rgi_reg,
                                                        version=rgi_version))
        rgidf = apply_rgi_fixes(rgidf, rgi_version, rgi_reg)
    elif isinstance(rgi_file, str):
        rgidf = gpd.read_file(rgi_file)
    else:
        rgidf = rgi_file

    if is_test:
        if test_ids is not None:
            try:
                rgidf = rgidf.loc[rgidf.RGIId.isin(test_ids)]
            except AttributeError:
                rgidf = rgidf.loc[rgidf.rgi_id.isin(test_ids)]
        else:
            rgidf = rgidf.sample(4, random_state=0)
    if len(rgidf) == 0:
        raise InvalidParamsError('Zero glaciers selected!')

    log.workflow(f'Starting the add-ons {products} for RGI reg: {rgi_reg}, '
                 f'border: {border}, {len(rgidf)} glaciers.')

    if start_from_dir is not None:
        from_tar = _level_dir(start_from_dir, rgi_version, border, '1')
        if not from_tar.is_dir():
            raise InvalidParamsError('Could not find the L1 glacier '
                                     f'directories in {from_tar}')
        gdirs = workflow.init_glacier_directories(
            rgidf, reset=True, force=True, from_tar=str(from_tar),
            prepro_base_url=start_base_url, prepro_border=border,
            prepro_rgi_version=rgi_version)
    else:
        gdirs = workflow.init_glacier_directories(
            rgidf, reset=True, force=True, from_prepro_level=1,
            prepro_border=border, prepro_rgi_version=rgi_version,
            prepro_base_url=start_base_url)

    for product in products:
        spec = ADDON_PRODUCTS[product]
        log.workflow(f'Add-on {product}...')
        workflow.execute_entity_task(_import(spec['task']), gdirs)

        out = Path(output_folder) / product
        workflow.execute_entity_task(_write_product_addon, gdirs,
                                     label=product,
                                     variables=spec['variables'],
                                     output_dir=str(out),
                                     source=spec['source'], version=version,
                                     preserve_totals=spec.get(
                                         'preserve_totals', True))
        utils.base_dir_to_tar(str(out))

        sum_dir = out / f'RGI{rgi_version}' / f'b_{border:03d}' / 'summary'
        utils.mkdir(sum_dir)
        _import(spec['statistics'])(
            gdirs, path=sum_dir / f'{product}_statistics_{rgi_reg}.csv')

    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    log.workflow('OGGM prepro_addons is done! Time needed: '
                 '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))


def parse_args(args):
    """Check input arguments and env variables"""

    description = ('Generate the add-ons (optional gridded data) for the '
                   'preprocessed OGGM glacier directories.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--map-border', type=int,
                        help='the size of the map border. Is required if '
                             '$OGGM_MAP_BORDER is not set.')
    parser.add_argument('--rgi-reg', type=str,
                        help='the rgi region to process. Is required if '
                             '$OGGM_RGI_REG is not set.')
    parser.add_argument('--rgi-version', type=str,
                        help='the RGI version to use. Defaults to the OGGM '
                             'default.')
    parser.add_argument('--products', nargs='+', required=True,
                        choices=list(ADDON_PRODUCTS),
                        help='the add-on products to make.')
    parser.add_argument('--start-base-url', type=str,
                        help='the base url of the tree holding the L1 '
                             'directories.')
    parser.add_argument('--start-from-dir', type=str,
                        help='like --start-base-url, for a tree on disk '
                             '(the folder holding the RGI{version}/b_{border}'
                             '/L1/ tree).')
    parser.add_argument('--version', type=str,
                        help='the version of the add-ons, recorded in their '
                             'manifests.')
    parser.add_argument('--output', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_OUTDIR.')
    parser.add_argument('--working-dir', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_WORKDIR.')
    parser.add_argument('--params-file', type=str,
                        help='path to the OGGM parameter file to use in place '
                             'of the default one.')
    parser.add_argument('--logging-level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')
    parser.add_argument('--test', nargs='?', const=True, default=False,
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    parser.add_argument('--test-ids', nargs='+',
                        help='if --test, specify the RGI ids to run separated '
                             'by a space (default: 4 randomly selected).')
    parser.add_argument('--disable-mp', nargs='?', const=True, default=False,
                        help='if you want to disable multiprocessing.')
    parser.add_argument('--override-params', type=json.loads, default=None)

    args = parser.parse_args(args)

    rgi_reg = args.rgi_reg or os.environ.get('OGGM_RGI_REG', None)
    if rgi_reg is None:
        raise InvalidParamsError('--rgi-reg is required!')
    rgi_reg = '{:02}'.format(int(rgi_reg))
    if rgi_reg not in ['{:02}'.format(int(r)) for r in range(1, 20)]:
        raise InvalidParamsError('--rgi-reg should range from 01 to 19!')

    border = args.map_border or os.environ.get('OGGM_MAP_BORDER', None)
    if border is None:
        raise InvalidParamsError('--map-border is required!')

    working_dir = args.working_dir or os.environ.get('OGGM_WORKDIR', '')
    output_folder = args.output or os.environ.get('OGGM_OUTDIR', '')

    return dict(rgi_version=args.rgi_version, rgi_reg=rgi_reg,
                border=int(border), products=args.products,
                output_folder=os.path.abspath(output_folder),
                working_dir=os.path.abspath(working_dir),
                start_base_url=args.start_base_url,
                start_from_dir=args.start_from_dir, version=args.version,
                params_file=args.params_file, is_test=args.test,
                test_ids=args.test_ids, disable_mp=args.disable_mp,
                logging_level=args.logging_level,
                override_params=args.override_params)


def main():
    """Script entry point"""

    run_prepro_addons(**parse_args(sys.argv[1:]))
