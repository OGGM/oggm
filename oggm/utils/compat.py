"""Compatibility and conversion wrappers between cumulative and
incremental prepro systems.

The main entry point is :func:`convert_prepro_to_deltas`, which converts
the previous cumulative per-level tar artifacts (each level tar
contains all lower levels' data) into the new incremental delta format.
Each level ships only the files it added or changed, plus an
``L{n}.manifest.json``, so clients can layer levels into one glacier
directory.

In the cumulative system, the default dataset set by DEFAULT_BASE_URL
spans two source URLs (levels 0-2 under ``L1-L2_files``, levels 3-5
under the spinup ``L3-L5_files`` tree), which is why per-level base URLs
are required. The dataset identity comes from an explicit tag rather
than a URL.
"""

import glob
import logging
import os
import shutil
import tempfile

import xarray as xr

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm.utils._workflow import (
    _finalize_merged_dir,
    base_dir_to_bundles,
    base_dir_to_tar,
    dataset_id_from_tag,
    gdir_to_archive,
    gdir_to_tar,
    robust_archive_extract,
    snapshot_gdir_state,
    write_level_manifest,
)

log = logging.getLogger(__name__)

# Shared data files. Divergence between the L0-L2 and L3-L5 source trees
# means the two trees were generated from different inputs.
_TREE_INVARIANTS = ("dem.tif", "glacier_grid.json", "dem_source.txt")

# basenames whose netCDF payload moves into the zarr store in v2
_NC_STORE_BASENAMES = ("gridded_data", "climate_historical", "gcm_data")


def _convert_pickles_to_zarr(gdir):
    """Rewrite a glacier directory's pickles into the zarr data store.

    One-way (not reversible): every ``.pkl`` that ``write_store`` can turn
    into a ``data_store.zarr/<group>`` is deleted afterwards, so the
    directory holds the same information in zarr form only. Suffixed
    variants (e.g. ``model_flowlines_dyn_melt_f_calib.pkl``) are handled by
    globbing each pickle BASENAME stem. Any pickle that ``write_store``
    cannot convert (it falls back to pickle) keeps its ``.pkl``, so no data
    is ever lost.

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to convert in place.
    """
    pkl_basenames = [
        k
        for k, v in cfg.BASENAMES.items()
        if isinstance(v, str) and v.endswith(".pkl")
    ]
    store_dir = os.path.join(gdir.dir, "data_store.zarr")
    for base in pkl_basenames:
        stem = cfg.BASENAMES[base][:-4]
        for fp in glob.glob(os.path.join(gdir.dir, f"{stem}*.pkl")):
            suffix = os.path.basename(fp)[len(stem) : -4]
            data = gdir.read_pickle(base, filesuffix=suffix)
            gdir.write_store(data, base, filesuffix=suffix)
            if os.path.isdir(os.path.join(store_dir, f"{base}{suffix}")):
                # zarr write succeeded; drop the now-redundant pickle
                os.remove(fp)
            # else write_store fell back to pickle: leave the .pkl in place


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
    become groups of ``data_store.zarr``
    - Shapefiles become geoparquet
    - Pickles converted to zarr with :func:`_convert_pickles_to_zarr`.
    - Model-run NetCDFs (``model_geometry*`` etc.) stay untouched.
    - Already-converted content is left alone.

    Parameters
    ----------
    gdir_or_dir : GlacierDirectory | str
        The glacier directory or its path, in the usual
        ``<base>/<region>/<subregion>/<rgi_id>`` layout.
    delete_originals : bool, default True
        If True, remove each ``.nc``/shapefile artifact once converted.

    Returns
    -------
    GlacierDirectory
        The converted v2 glacier directory.
    """
    gdir = _as_gdir(gdir_or_dir)
    _convert_pickles_to_zarr(gdir)

    for base in _NC_STORE_BASENAMES:
        stem = cfg.BASENAMES[base][:-3]  # strip .nc
        for fp in sorted(glob.glob(os.path.join(gdir.dir, f"{stem}*.nc"))):
            suffix = os.path.basename(fp)[len(stem) : -3]
            with xr.open_dataset(fp, decode_cf=False) as ds:
                gdir.write_group(ds.load(), base, filesuffix=suffix, mode="w")
            if delete_originals:
                os.remove(fp)

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
    - gridded/climate/gcm zarr groups become ``.nc`` files again
    - geoparquet become tarred shapefiles.
    - Pickle-era store groups are left in the store as ``read_store``
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
    store_dir = os.path.join(gdir.dir, "data_store.zarr")

    for base in _NC_STORE_BASENAMES:
        for group_dir in sorted(glob.glob(os.path.join(store_dir, f"{base}*"))):
            suffix = os.path.basename(group_dir)[len(base) :]
            with gdir.open_group(
                base, filesuffix=suffix, decode_cf=False
            ) as ds:
                ds = ds.load()
            gdir.delete_group(base, filesuffix=suffix)
            ds.to_netcdf(gdir.get_filepath(base, filesuffix=suffix))

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


def convert_prepro_to_v2_artifacts(
    delta_tree: str,
    rgi_ids: list[str],
    output_dir: str,
    dataset_tag: str,
    border: int = 80,
    rgi_version: str = "62",
    workdir: str | None = None,
    max_level: int = 5,
) -> str:
    """Convert a local v1 delta tree into v2 artifacts.

    Composes from an existing local delta-format tree produced by
    :func:`convert_prepro_to_deltas`.

    Per glacier, levels are layered in ascending order. Each
    materialisations restarts the layering, with deltas stacking on top.
    L5 remains standalone standalone converts in isolation. Each layered
    state is converted with :func:`convert_gdir_to_v2`, diffed against the
    previous converted state, and repackaged as stored zip bundles with
    ``format_version: 2`` manifests that carry over each v1 level's
    kind/requires/includes.

    Parameters
    ----------
    delta_tree : str
        Root of the v1 delta tree (the directory holding
        ``RGI{v}/b_{bbb}/L{n}/...``).
    rgi_ids : list[str]
        Glaciers to convert.
    output_dir : str
        Root of the v2 output tree
    dataset_tag : str
        Tag of the converted dataset. This is hashed into the new
        ``dataset_id`` as the v2 artifacts are a distinct logical
        dataset from their v1 source.
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
    levels = sorted(
        int(d[1:])
        for d in os.listdir(tree_root)
        if d.startswith("L") and d[1:].isdigit() and int(d[1:]) <= max_level
    )
    if not levels:
        raise InvalidParamsError(f"No level dirs under {tree_root}")

    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix="oggm_v2_convert_")
    dataset_id = dataset_id_from_tag(dataset_tag, border, rgi_version)

    try:
        for rid in rgi_ids:
            region = rid[:-6]
            bundle = f"{region}.{rid[-5:-2]}"
            cum_base = os.path.join(workdir, "cum", rid)
            prev_state = None
            for lvl in levels:
                bundle_tar = os.path.join(
                    tree_root, f"L{lvl}", region, f"{bundle}.tar"
                )
                if not os.path.isfile(bundle_tar):
                    continue
                v1m = _peek_level_manifest(bundle_tar, rid, lvl)
                if v1m is None:
                    log.warning(
                        "(%s) L%d has no manifest (legacy cumulative "
                        "bundle?): skipped.",
                        rid,
                        lvl,
                    )
                    continue
                kind = v1m.get("kind", "delta")
                requires = v1m.get("requires") or []
                includes = v1m.get("includes_levels") or [lvl]
                member = os.path.join(
                    tree_root, f"L{lvl}", region, bundle, rid + ".tar.gz"
                )

                if kind == "standalone":
                    base = os.path.join(workdir, f"sa_L{lvl}", rid)
                else:
                    base = cum_base
                    if not requires and os.path.isdir(base):
                        # materialisations restart the layering
                        shutil.rmtree(base)
                gdir_dir = os.path.join(base, region, rid[:-3], rid)
                os.makedirs(os.path.dirname(gdir_dir), exist_ok=True)
                robust_archive_extract(member, gdir_dir)
                # replaced below by the v2 manifest, removing it first
                # keeps _finalize_merged_dir's compat check on v2 only
                os.remove(os.path.join(gdir_dir, f"L{lvl}.manifest.json"))
                _finalize_merged_dir(gdir_dir)

                gdir = GlacierDirectory(rid, base_dir=base)
                convert_gdir_to_v2(gdir)
                full = kind == "standalone" or not requires
                _, changed = write_level_manifest(
                    gdir,
                    level=lvl,
                    prev_state={} if full else (prev_state or {}),
                    requires=requires,
                    includes_levels=includes,
                    kind=kind,
                    dataset_tag=dataset_tag,
                    dataset_id=dataset_id,
                    border=border,
                    rgi_version=rgi_version,
                    format_version=2,
                )
                stage_dir = os.path.join(out_root, f"L{lvl}")
                gdir_to_archive.unwrapped(
                    gdir,
                    base_dir=stage_dir,
                    delete=False,
                    include=None if full else changed,
                    fmt="zip",
                )
                if kind != "standalone":
                    prev_state = snapshot_gdir_state(gdir.dir)
        for lvl in levels:
            stage_dir = os.path.join(out_root, f"L{lvl}")
            if os.path.isdir(stage_dir):
                base_dir_to_bundles(stage_dir, delete=True, fmt="zip")
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    return out_root


def convert_prepro_to_deltas(
    rgi_ids: list[str],
    base_urls: dict[int, str],
    border: int,
    rgi_version: str,
    workdir: str,
    output_dir: str,
    dataset_tag: str,
    max_level: int = 5,
    convert_to_zarr: bool = False,
):
    """Convert cumulative prepro artifacts into per-level delta bundles.

    You can use this to convert entire RGI regions ready for upload
    directly to the cluster.

    Downloads each available level of the given glaciers into isolated
    working directories, diffs successive levels, and writes a
    delta-format tree under ``output_dir``:
    ``{output_dir}/RGI{rgi_version}/b_{border:03d}/L{n}/{region}/{bundle}.tar``.

    Artifact kinds: L0 and L3 are standalone (self-sufficient,
    ``requires=[]``), intermediate levels are deltas against the level
    below, and L5 is a standalone bundle.

    Parameters
    ----------
    rgi_ids : list[str]
        Glaciers to convert.
    base_urls : dict[int, str]
        Per-level source base URL. One logical dataset can be served
        from several URLs, e.g. L0-L2 from the ``L1-L2_files`` tree and
        L3-L5 from the spinup tree.
    border : int
        Map border of the source dataset.
    rgi_version : str
        RGI version of the source dataset.
    workdir : str
        Scratch directory for the per-level downloads.
    output_dir : str
        Root of the delta-format output tree.
    dataset_tag : str
        Explicit label identifying the logical dataset. This is hashed
        with border and RGI version into the manifest's ``dataset_id``.
        Must **not** be a source URL.
    max_level : int, default=5
        Convert levels up to and including this one.
    convert_to_zarr : bool, default=False
        If True, rewrite each glacier's pickle files into its
        ``data_store.zarr`` store (and delete the pickles) before
        tarring, so the output tree ships zarr instead of pickles.
        This is a one-way process, but the output holds the same
        information as the input pickles.

    Returns
    -------
    str
        The output level-tree root,
        ``{output_dir}/RGI{rgi_version}/b_{border:03d}``.
    """
    # Import here to avoid circular import
    from oggm import workflow

    levels = sorted(lvl for lvl in base_urls if lvl <= max_level)
    if not levels:
        raise InvalidParamsError("base_urls contains no level <= max_level")
    lowest = levels[0]
    dataset_id = dataset_id_from_tag(dataset_tag, border, rgi_version)
    out_root = os.path.join(
        output_dir, f"RGI{rgi_version}", f"b_{int(border):03d}"
    )

    prev_working_dir = cfg.PATHS.get("working_dir", "")
    prev_states = {}
    try:
        for lvl in levels:
            level_wdir = os.path.join(workdir, f"L{lvl}")
            os.makedirs(level_wdir, exist_ok=True)
            cfg.PATHS["working_dir"] = level_wdir
            gdirs = workflow.init_glacier_directories(
                rgi_ids,
                from_prepro_level=lvl,
                prepro_border=border,
                prepro_rgi_version=rgi_version,
                prepro_base_url=base_urls[lvl],
            )

            stage_dir = os.path.join(out_root, f"L{lvl}")
            for gdir in gdirs:
                if convert_to_zarr:
                    _convert_pickles_to_zarr(gdir)
                include = _write_artifact_manifest(
                    gdir=gdir,
                    level=lvl,
                    lowest=lowest,
                    prev_state=prev_states.get(gdir.rgi_id),
                    dataset_tag=dataset_tag,
                    dataset_id=dataset_id,
                    border=border,
                    rgi_version=rgi_version,
                )
                gdir_to_tar.unwrapped(
                    gdir, base_dir=stage_dir, delete=False, include=include
                )
                prev_states[gdir.rgi_id] = snapshot_gdir_state(gdir.dir)
            base_dir_to_tar(stage_dir, delete=True)
    finally:
        cfg.PATHS["working_dir"] = prev_working_dir

    return out_root


def _write_artifact_manifest(
    gdir,
    level: int,
    lowest: int,
    prev_state: dict,
    border: int,
    rgi_version: str,
    dataset_tag: str,
    dataset_id: str = "",
):
    """Write the level manifest and return the tar include list.

    L3 is published as a standalone materialisation of L0-L3, as it's
    the entry point for the spinup tree. This means L3/L4/L5 stay within
    2-3 requests.

    Also ensures that separately generated source trees agree on shared
    data files set by _TREE_INVARIANTS.

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to snapshot.
    level : int
        The prepro level being written.
    lowest : int
        The lowest level being converted.
    prev_state : dict or None
        The previous level's snapshot of the glacier directory, or None
        if this is the lowest level.
    dataset_id : str
        The dataset identity, hashed from the dataset tag, border, and
        RGI version.
    dataset_tag : str
        The dataset tag, identifying the logical dataset.
    border : int
        The map border of the source dataset.
    rgi_version : str
        The RGI version of the source dataset.

    Returns
    -------
    None or list[str]
        None (full tar) for materialisations and the standalone L5
        bundle, or a list of changed paths for delta levels.
    """

    if not dataset_id:
        dataset_id = dataset_id_from_tag(dataset_tag, border, rgi_version)

    common = dict(
        dataset_id=dataset_id,
        dataset_tag=dataset_tag,
        border=border,
        rgi_version=rgi_version,
    )
    if level == 5:
        write_level_manifest(
            gdir,
            level=5,
            prev_state={},
            requires=[],
            includes_levels=[5],
            kind="standalone",
            **common,
        )
        return None
    if level == lowest:
        write_level_manifest(
            gdir,
            level=level,
            prev_state={},
            requires=[],
            includes_levels=list(range(lowest, level + 1)),
            **common,
        )
        return None
    if level == 3:
        # L3 is a materialisation of L0-L3 (see docstring)
        if prev_state is not None:
            state = snapshot_gdir_state(gdir)
            diverged = [
                f
                for f in _TREE_INVARIANTS
                if f in prev_state and f in state and prev_state[f] != state[f]
            ]
            if diverged:
                log.warning(
                    "(%s) the L%d source tree disagrees with the level "
                    "below on %s: the L0-L%d artifacts belong to a "
                    "different dataset generation than the L3 materialisation.",
                    gdir.rgi_id,
                    level,
                    diverged,
                    level - 1,
                )
        write_level_manifest(
            gdir,
            level=3,
            prev_state={},
            requires=[],
            includes_levels=[0, 1, 2, 3],
            **common,
        )
        return None
    if prev_state is None:
        raise InvalidParamsError(
            f"Cannot build a delta for level {level} without the level "
            "below it in base_urls."
        )
    _, changed = write_level_manifest(
        gdir,
        level=level,
        prev_state=prev_state,
        requires=list(range(lowest, level)),
        **common,
    )
    return changed
