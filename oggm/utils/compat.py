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

import logging
import os

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm.utils._workflow import (
    base_dir_to_tar,
    dataset_id_from_tag,
    gdir_to_tar,
    snapshot_gdir_state,
    write_level_manifest,
)

log = logging.getLogger(__name__)

# Shared data files. Divergence between the L0-L2 and L3-L5 source trees
# means the two trees were generated from different inputs.
_TREE_INVARIANTS = ("dem.tif", "glacier_grid.json", "dem_source.txt")


def convert_prepro_to_deltas(
    rgi_ids: list[str],
    base_urls: dict[int, str],
    border: int,
    rgi_version: str,
    workdir: str,
    output_dir: str,
    dataset_tag: str,
    max_level: int = 5,
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

    L3 is published as a standalone rollup of L0-L3, as it's the entry
    point for the spinup tree. This means L3/L4/L5 stay within 2-3
    requests.

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
        None (full tar) for rollups and the standalone L5 bundle, or a
        list of changed paths for delta levels.
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
        # L3 is a rollup of L0-L3 (see docstring)
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
                    "different dataset generation than the L3 rollup.",
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
