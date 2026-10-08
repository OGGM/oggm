"""Compatibility and conversion wrappers between legacy and new glacier
directory formats.

The main entry point is :func:`convert_prepro_to_deltas`, which converts
the previous cumulative per-level tar artifacts (each level tar
contains all lower levels' data) into the new incremental delta format.
Each level ships only the files it added, changed, or removed, plus a
``L{n}.manifest.json`` naming its parent, so clients can layer levels
into one glacier directory.

One call converts one source tree. In the cumulative system, the
default dataset spans two trees (levels 0-2 under ``L1-L2_files``,
levels 3-5 under the spinup ``L3-L5_files`` tree), so it takes two
calls: the second links its first level to the converted, published
copy of the first tree through ``parent_base_url``.
"""

import glob
import logging
import os
from pathlib import Path

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm.utils._workflow import (
    base_dir_to_tar,
    gdir_to_tar,
    snapshot_gdir_state,
    write_level_manifest,
)

log = logging.getLogger(__name__)

# Shared data files. Divergence between the L0-L2 and L3-L5 source trees
# means the two trees were generated from different inputs.
_TREE_INVARIANTS = ("dem.tif", "glacier_grid.json", "dem_source.txt")


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
            if npz_fp.is_file() and delete:
                # npz write succeeded, drop the now-redundant pickle
                os.remove(fp)
            else:
                pass  # fell back to pickle so leave .pkl in place


def convert_prepro_to_deltas(
    rgi_ids: list[str],
    src_base_url: str,
    levels: list[int],
    border: int,
    rgi_version: str,
    workdir: str,
    output_dir: str,
    parent_base_url: str | None = None,
    convert_to_npz: bool = False,
):
    """Convert one cumulative prepro tree into per-level delta bundles.

    You can use this to convert entire RGI regions ready for upload
    directly to the cluster.

    Downloads each level of the given glaciers into isolated working
    directories, diffs successive levels, and writes a delta-format tree
    under ``output_dir``:
    ``{output_dir}/RGI{rgi_version}/b_{border:03d}/L{n}/{region}/{bundle}.tar``.

    Artifact kinds: L0 is the root delta, and every other level is a
    delta on the level converted before it. The first level above L0 is
    a delta on the level below it in ``parent_base_url`` if given, else a
    materialisation. L5 is a standalone bundle, with L4 as its
    provenance parent.

    Parameters
    ----------
    rgi_ids : list[str]
        Glaciers to convert.
    src_base_url : str
        Base URL of the cumulative source tree, e.g. the ``L1-L2_files``
        or the spinup ``L3-L5_files`` tree.
    levels : list[int]
        Levels to convert, e.g. ``[0, 1, 2]`` or ``[3, 4, 5]``.
    border : int
        Map border of the source dataset.
    rgi_version : str
        RGI version of the source dataset.
    workdir : str
        Scratch directory for the per-level downloads.
    output_dir : str
        Root of the delta-format output tree.
    parent_base_url : str | None, optional
        Base URL of an already converted, published tree holding the
        level below ``levels[0]``. Its glacier directories are the
        baseline of the first level, so the output always sits on the
        converted tree, never on the cumulative one.
    convert_to_npz : bool, default=False
        If True, rewrite each glacier's pickle files into its
        ``data_store`` store (and delete the pickles) before tarring, so
        the output tree ships npz instead of pickles. This is a one-way
        process, but the output holds the same information as the input
        pickles.

    Returns
    -------
    str
        The output level-tree root,
        ``{output_dir}/RGI{rgi_version}/b_{border:03d}``.
    """
    # Import here to avoid circular import
    from oggm import workflow

    levels = sorted(int(lvl) for lvl in levels)
    if not levels:
        raise InvalidParamsError("levels is empty")
    out_root = os.path.join(
        output_dir, f"RGI{rgi_version}", f"b_{int(border):03d}"
    )

    def init(level, base_url, name):
        level_wdir = os.path.join(workdir, name)
        os.makedirs(level_wdir, exist_ok=True)
        cfg.PATHS["working_dir"] = level_wdir
        return workflow.init_glacier_directories(
            rgi_ids,
            from_prepro_level=level,
            prepro_border=border,
            prepro_rgi_version=rgi_version,
            prepro_base_url=base_url,
        )

    prev_working_dir = cfg.PATHS.get("working_dir", "")
    # Per glacier: the baseline's state, directory, and label
    prev = {}
    parent_url = parent_base_url
    try:
        if parent_base_url is not None:
            below = levels[0] - 1
            for gdir in init(below, parent_base_url, f"parent_L{below}"):
                prev[gdir.rgi_id] = (
                    snapshot_gdir_state(gdir.dir),
                    gdir.dir,
                    f"L{below}",
                )
        for lvl in levels:
            stage_dir = os.path.join(out_root, f"L{lvl}")
            for gdir in init(lvl, src_base_url, f"L{lvl}"):
                if convert_to_npz:
                    convert_pickles_to_npz(gdir)
                include = _write_artifact_manifest(
                    gdir,
                    f"L{lvl}",
                    prev.get(gdir.rgi_id),
                    parent_url,
                    border=border,
                    rgi_version=rgi_version,
                )
                gdir_to_tar.unwrapped(
                    gdir, base_dir=stage_dir, delete=False, include=include
                )
                prev[gdir.rgi_id] = (
                    snapshot_gdir_state(gdir.dir),
                    gdir.dir,
                    f"L{lvl}",
                )
            base_dir_to_tar(stage_dir, delete=True)
            parent_url = None  # later parents are in this tree
    finally:
        cfg.PATHS["working_dir"] = prev_working_dir

    return out_root


def _write_artifact_manifest(
    gdir,
    label: str,
    prev: tuple | None,
    parent_base_url: str | None,
    border: int,
    rgi_version: str,
) -> list[str]:
    """Write the level manifest and return the tar include list.

    Also warns when the baseline and this level disagree on the shared
    data files set by _TREE_INVARIANTS.

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to snapshot.
    label : str
        The label being written, e.g. ``"L3"``.
    prev : tuple or None
        The baseline's ``(state, directory, label)``, or None if there is
        nothing below this level.
    parent_base_url : str or None
        Base URL of the tree holding the baseline, or None if it is this
        tree.
    border : int
        The map border of the source dataset.
    rgi_version : str
        The RGI version of the source dataset.

    Returns
    -------
    list[str]
        The changed paths and the manifest, for ``gdir_to_tar(include=...)``.
    """
    common = dict(border=border, rgi_version=rgi_version)
    if prev is None:
        # L0 is the root; anything else with nothing below ships every file
        kind = "delta" if label == "L0" else "materialisation"
        if label == "L5":
            kind = "standalone"
        _, changed = write_level_manifest(gdir, label, {}, kind=kind, **common)
        return changed

    prev_state, prev_dir, prev_label = prev
    state = snapshot_gdir_state(gdir)
    diverged = [
        f
        for f in _TREE_INVARIANTS
        if f in prev_state and f in state and prev_state[f] != state[f]
    ]
    if diverged:
        log.warning(
            "(%s) the %s source tree disagrees with %s on %s: they belong "
            "to different dataset generations.",
            gdir.rgi_id,
            label,
            prev_label,
            diverged,
        )
    _, changed = write_level_manifest(
        gdir,
        label,
        prev_state,
        parent=prev_label,
        parent_base_url=parent_base_url,
        parent_dir=prev_dir,
        kind="standalone" if label == "L5" else "delta",
        **common,
    )
    return changed
