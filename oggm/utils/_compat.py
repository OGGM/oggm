"""Compatibility and conversion wrappers between legacy and glacier
directory formats

The main entry point is :func:`convert_prepro_to_npz`, which converts
pickles in glacier directories into npz and repackages them.
"""

import glob
import logging
import os
from pathlib import Path

from oggm import cfg

log = logging.getLogger(__name__)


def convert_pickles_to_npz(gdir, delete: bool = True):
    """Rewrite a glacier directory's pickles into npz.

    One-way (not reversible): every pickle that ``write_store`` can turn
    into a ``data_store/<data>.npz`` is deleted afterwards, so the
    directory holds the same information in npz form only.
    Suffixed variants (e.g. ``model_flowlines_dyn_melt_f_calib.pkl``)
    are handled by globbing each pickle BASENAME stem. Any pickle that
    ``write_store`` cannot convert (it falls back to pickle) keeps its
    ``.pkl``, so no data is ever lost.

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to convert in place.
    delete : bool, default True
        If True (recommended), delete the original pickles after
        conversion. If False, keep them around for comparison. This is
        irreversible, the directory will hold the same information in
        npz form only.
    """
    pkl_basenames = [
        k
        for k, v in cfg.BASENAMES.items()
        if isinstance(v, str) and v.endswith(".pkl")
    ]

    store_dir = Path(gdir.dir) / "data_store"
    for base in pkl_basenames:
        stem = cfg.BASENAMES[base][:-4]
        # we want all possible pickles
        for fp in glob.glob(os.path.join(Path(gdir.dir), f"{stem}*.pkl")):
            suffix = os.path.basename(fp)[len(stem) : -4]
            data = gdir._read_pickle(base, filesuffix=suffix)
            gdir.write_store(data, base, filesuffix=suffix)
            npz_fp = store_dir / f"{base}{suffix}.npz"
            if os.path.isfile(npz_fp) and delete:
                # npz write succeeded, drop the now-redundant pickle
                os.remove(fp)
            else:
                pass  # fell back to pickle so leave .pkl in place
