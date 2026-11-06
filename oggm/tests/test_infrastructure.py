"""Outage labels on test failures, from the hooks in conftest.py.

This module tests the labeling of test failures due to infrastructure outages.
"""

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.test_env("utils")

URL = "https://cluster.invalid.com/data.nc"

OUTAGE = f"""
import pytest
from oggm.exceptions import DownloadOutageError

def outage():
    return DownloadOutageError({URL!r}, 3, ConnectionError("refused"))
"""


@pytest.fixture
def run(pytester, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    pytester.makeconftest(
        "from oggm.tests.conftest import pytest_runtest_setup, "
        "pytest_runtest_makereport, pytest_terminal_summary, "
        "pytest_sessionfinish"
    )

    def _run(body):
        pytester.makepyfile(OUTAGE + body)
        return pytester.runpytest()

    return _run


def test_chained_outage_is_labelled(run):
    res = run("""
def test_task():
    raise RuntimeError("task failed") from outage()
""")
    assert res.ret == 75
    res.stdout.fnmatch_lines(["*INFRA OUTAGE*", f"*{URL}*3 attempts*"])


def test_one_logic_failure_keeps_exit_1(run):
    res = run("""
def test_task():
    raise RuntimeError("task failed") from outage()

def test_logic():
    assert False
""")
    assert res.ret == 1
    res.stdout.fnmatch_lines(["*= INFRA OUTAGE =*", "*::test_task: *"])
    assert "::test_logic: " not in res.stdout.str()


def test_fixture_outage_is_labelled(run):
    res = run("""
@pytest.fixture
def gdir():
    raise outage()

def test_task(gdir):
    pass
""")
    assert res.ret == 75
    res.stdout.fnmatch_lines(["*INFRA OUTAGE*"])


def test_swallowed_outage_is_probable(run):
    res = run(f"""
from oggm.utils import _downloads

def test_task():
    _downloads.OUTAGES.append(({URL!r}, str(outage())))
    raise ValueError("missing output")
""")
    assert res.ret == 75
    res.stdout.fnmatch_lines(["*INFRA OUTAGE*", f"*probable*{URL}*"])


TWO_OUTAGES = f"""
from oggm.utils import _downloads

def test_task():
    raise RuntimeError("task failed") from outage()

def test_swallowed():
    _downloads.OUTAGES.append(({URL!r}, str(outage())))
    raise ValueError("missing output")
"""


def test_local_summary_section(run):
    res = run(TWO_OUTAGES)
    res.stdout.fnmatch_lines(
        [
            "*= INFRA OUTAGE =*",
            f"*::test_task: {URL}: 3 attempts*",
            f"*::test_swallowed: probable: {URL}*",
        ]
    )
    assert "::error" not in res.stdout.str()


def test_github_annotations_and_step_summary(run, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    res = run(TWO_OUTAGES)
    res.stdout.fnmatch_lines(
        [
            f"::error title=Download outage::*::test_task: {URL}",
            f"::error title=Download outage::*::test_swallowed: {URL}",
        ]
    )
    rows = [
        line
        for line in summary.read_text().splitlines()
        if line.startswith("| ")
    ]
    assert len(rows) == 3  # header plus one row per failure
    assert rows[1].endswith("| no |")
    assert rows[2].endswith("| yes |")


def test_green_run_untouched(run, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    res = run("""
def test_ok():
    pass
""")
    assert res.ret == 0
    assert "INFRA OUTAGE" not in res.stdout.str()
    assert not summary.exists()
