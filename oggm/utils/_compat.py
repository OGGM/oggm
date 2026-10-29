"""Compatibility and conversion wrappers between legacy and glacier
directory formats.

The main entry point is :func:`convert_pickles_to_npz`, which converts
the pickles of a glacier directory into npz.
"""

import glob
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile

import xarray as xr

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm.utils._workflow import (
    base_dir_to_bundles,
    gdir_to_archive,
    robust_archive_extract,
    snapshot_gdir_state,
    write_level_manifest,
)

log = logging.getLogger(__name__)

# dataset-group basenames, rewritten compressed by write_group in v2
_NC_STORE_BASENAMES = ("gridded_data", "climate_historical", "gcm_data")


def convert_pickles_to_npz(gdir, delete: bool = True):
    """Rewrite a glacier directory's pickles into npz.

    One-way (not reversible): every pickle that ``write_store`` can turn
    into a ``<data>.npz`` is deleted afterwards, so the directory holds
    the same information in npz form only. Suffixed variants (e.g.
    ``model_flowlines_dyn_melt_f_calib.pkl``) are handled by globbing
    each pickle BASENAME stem. Any pickle that ``write_store`` cannot
    convert (it falls back to pickle) keeps its ``.pkl``, so no data is
    ever lost.

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

    for base in pkl_basenames:
        stem = cfg.BASENAMES[base][:-4]
        # we want all possible pickles
        for fp in glob.glob(os.path.join(Path(gdir.dir), f"{stem}*.pkl")):
            suffix = os.path.basename(fp)[len(stem) : -4]
            data = gdir._read_pickle(base, filesuffix=suffix)
            gdir.write_store(data, base, filesuffix=suffix)
            npz_fp = gdir.get_store_filepath(base, suffix)
            if os.path.isfile(npz_fp) and delete:
                # npz write succeeded, drop the now-redundant pickle
                os.remove(fp)
            else:
                pass  # fell back to pickle so leave .pkl in place


def _as_gdir(gdir_or_dir):
    """A GlacierDirectory for either a gdir or a per_glacier-layout path."""
    if hasattr(gdir_or_dir, "dir"):
        return gdir_or_dir
    from oggm import GlacierDirectory

    d = os.path.normpath(gdir_or_dir)
    # <base_dir>/<region>/<subregion>/<rgi_id>
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(d)))
    return GlacierDirectory(os.path.basename(d), base_dir=base_dir)


def convert_gdir_to_v2(gdir_or_dir, delete_originals: bool = True):
    """Convert a v1 glacier directory to the v2 payload formats in place.

    - ``gridded_data*.nc`` / ``climate_historical*.nc`` / ``gcm_data*.nc``
      are rewritten in place through ``write_group`` (compressed).
    - Pickles become npz store groups with :func:`convert_pickles_to_npz`.
    - Model-run NetCDFs (``model_geometry*`` etc.) stay untouched. TODO: rewrite these too.
    - Already-converted content is left alone.
    - Shapefiles become geoparquet.

    Parameters
    ----------
    gdir_or_dir : GlacierDirectory | str
        The glacier directory or its path, in the usual
        ``<base>/<region>/<subregion>/<rgi_id>`` layout.
    delete_originals : bool, default True
        If True, remove each shapefile artefact once converted.

    Returns
    -------
    GlacierDirectory
        The converted v2 glacier directory.
    """
    gdir = _as_gdir(gdir_or_dir)
    convert_pickles_to_npz(gdir)

    for base in _NC_STORE_BASENAMES:
        stem = cfg.BASENAMES[base][:-3]  # strip .nc
        for fp in sorted(glob.glob(os.path.join(gdir.dir, f"{stem}*.nc"))):
            suffix = os.path.basename(fp)[len(stem) : -3]
            with xr.open_dataset(fp, decode_cf=False) as ds:
                ds = ds.load()
            gdir.write_group(ds, base, filesuffix=suffix, mode="w")

    shp_basenames = [
        k
        for k, v in cfg.BASENAMES.items()
        if isinstance(v, str) and v.endswith(".shp")
    ]
    for base in shp_basenames:
        stem = cfg.BASENAMES[base][:-4]  # strip .shp
        found = {}
        for fp in glob.glob(os.path.join(gdir.dir, f"{stem}*.tar.gz")):
            found[os.path.basename(fp)[len(stem) : -7]] = fp
        for fp in glob.glob(os.path.join(gdir.dir, f"{stem}*.shp")):
            found.setdefault(os.path.basename(fp)[len(stem) : -4], fp)
        for suffix in sorted(found):
            gdf = gdir.read_shapefile(base, filesuffix=suffix)
            gdir.write_shapefile(gdf, base, filesuffix=suffix)
            if delete_originals:
                for old in glob.glob(
                    os.path.join(gdir.dir, f"{stem}{suffix}.*")
                ):
                    if not old.endswith(".parquet"):
                        os.remove(old)
    return gdir


def convert_gdir_to_v16(gdir_or_dir):
    """Convert a v2 glacier directory in place to the v1.6 format.

    The reverse of :func:`convert_gdir_to_v2`.
    - geoparquet become tarred shapefiles.
    - gridded/climate/gcm groups are already ``.nc`` files and stay.
    - npz store groups are left in the store as ``read_store``
      reads them either way.

    Parameters
    ----------
    gdir_or_dir : GlacierDirectory | str
        The glacier directory or its path to convert.

    Returns
    -------
    GlacierDirectory
        The converted v1 glacier directory.
    """
    from oggm.utils._workflow import _write_shape_to_disk

    gdir = _as_gdir(gdir_or_dir)

    shp_basenames = [
        k
        for k, v in cfg.BASENAMES.items()
        if isinstance(v, str) and v.endswith(".shp")
    ]
    for base in shp_basenames:
        stem = cfg.BASENAMES[base][:-4]
        for fp in sorted(glob.glob(os.path.join(gdir.dir, f"{stem}*.parquet"))):
            suffix = os.path.basename(fp)[len(stem) : -8]
            gdf = gdir.read_shapefile(base, filesuffix=suffix)
            # v1.6 dirs must not keep the parquet (reads are
            # parquet-first)
            os.remove(fp)
            _write_shape_to_disk(
                gdf,
                gdir.get_filepath(base, filesuffix=suffix),
                to_tar=cfg.PARAMS["use_tar_shapefiles"],
            )
    return gdir


def convert_prepro_to_v2_artefacts(
    delta_tree: str,
    rgi_ids: list[str],
    output_dir: str,
    border: int = 80,
    rgi_version: str = "62",
    workdir: str | None = None,
    max_level: int = 5,
) -> str:
    """Convert a local v1 parent-chain tree into v2 artefacts.

    Per glacier, each level's v1 member is layered onto the converted
    state of its parent, without removed files, and converted with
    :func:`convert_gdir_to_v2`. A delta only ships changes against its
    converted parent, so if a file in a v1 level wasn't modified, it
    won't be in its v2 delta. A standalone (the L1 root, or L5) ships
    every file. The v2 manifests keep each v1 level's kind and parent
    label, with ids of their own.

    .. note::
        Parents are looked up in the same tree. Convert a v1 tree whose
        parents live at another URL from a local copy holding every
        level.

    Parameters
    ----------
    delta_tree : str
        Root of the v1 tree (the directory holding
        ``RGI{v}/b_{bbb}/L{n}/...``).
    rgi_ids : list[str]
        Glaciers to convert.
    output_dir : str
        Root of the v2 output tree.
    border : int, default 80
        Map border of the dataset.
    rgi_version : str, default '62'
        RGI version of the dataset.
    workdir : str | None
        Scratch directory, defaults to a temporary one.
    max_level : int, default 5
        Convert levels up to and including this one.

    Returns
    -------
    str
        The output level-tree root,
        ``{output_dir}/RGI{rgi_version}/b_{border:03d}``.
    """
    from oggm import GlacierDirectory
    from oggm.workflow import _peek_level_manifest

    tree_root = os.path.join(
        delta_tree, f"RGI{rgi_version}", f"b_{int(border):03d}"
    )
    if not os.path.isdir(tree_root):
        raise InvalidParamsError(f"No delta tree at {tree_root}")
    out_root = os.path.join(
        output_dir, f"RGI{rgi_version}", f"b_{int(border):03d}"
    )
    # don't convert freeform labels (e.g. L3a) as they're never published
    labels = sorted(
        (
            d
            for d in os.listdir(tree_root)
            if re.fullmatch(r"L\d", d) and int(d[1:]) <= max_level
        ),
        key=lambda d: int(d[1:]),
    )
    if not labels:
        raise InvalidParamsError(f"No level dirs under {tree_root}")

    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix="oggm_v2_convert_")

    try:
        for rid in rgi_ids:
            region = rid[:-6]
            bundle = f"{region}.{rid[-5:-2]}"
            chain_base = os.path.join(workdir, "chain", rid)
            chain_dir = os.path.join(chain_base, region, rid[:-3], rid)
            prev_state = {}
            for label in labels:
                bundle_tar = os.path.join(
                    tree_root, label, region, f"{bundle}.tar"
                )
                if not os.path.isfile(bundle_tar):
                    continue
                v1m = _peek_level_manifest(bundle_tar, rid, label)
                if v1m is None:
                    log.warning(
                        "(%s) %s has no manifest (legacy cumulative "
                        "bundle?): skipped.",
                        rid,
                        label,
                    )
                    continue
                kind = v1m["kind"]
                parent = (v1m.get("parent") or {}).get("label")
                member = os.path.join(
                    tree_root, label, region, bundle, rid + ".tar.gz"
                )

                base = chain_base
                if kind == "standalone" and parent:
                    # e.g. L5: a side branch, converted in isolation
                    base = os.path.join(workdir, label, rid)
                gdir_dir = os.path.join(base, region, rid[:-3], rid)
                if kind == "standalone" and os.path.isdir(gdir_dir):
                    shutil.rmtree(gdir_dir)
                os.makedirs(os.path.dirname(gdir_dir), exist_ok=True)
                robust_archive_extract(member, gdir_dir)
                # The v1 parent ids don't match the converted parents so
                # removed files are deleted here
                for rel in v1m["files"].get("removed", []):
                    fp = os.path.join(gdir_dir, rel)
                    if os.path.isfile(fp):
                        os.remove(fp)
                os.remove(os.path.join(gdir_dir, f"{label}.manifest.json"))

                gdir = GlacierDirectory(rid, base_dir=base)
                convert_gdir_to_v2(gdir)
                _, changed = write_level_manifest(
                    gdir,
                    label,
                    prev_state,
                    parent=parent,
                    parent_dir=chain_dir,
                    kind=kind,
                    border=border,
                    rgi_version=rgi_version,
                )
                gdir_to_archive.unwrapped(
                    gdir,
                    base_dir=os.path.join(out_root, label),
                    delete=False,
                    include=changed if kind == "delta" else None,
                    fmt="zip",
                )
                if base == chain_base:
                    prev_state = snapshot_gdir_state(gdir.dir)
        for label in labels:
            stage_dir = os.path.join(out_root, label)
            if os.path.isdir(stage_dir):
                base_dir_to_bundles(stage_dir, delete=True, fmt="zip")
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    return out_root
