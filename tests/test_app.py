"""Smoke test of the Streamlit dashboard (Stage 10).

Runs every page for every horizon without a browser and checks that no page
raises an error. Needs the demo bundle (`python -m app.build_demo`); the test
is skipped when the bundle is missing (for example in a fresh clone).
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "streamlit_app.py"
BUNDLE = REPO / "app" / "demo" / "manifest.json"
PAGES = ["Overview", "Forecast vs actual", "Explain a forecast", "Fault flags",
         "Efficiency", "Generalisation"]

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")


def run_app(page, horizon, station=None):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["page"] = page
    app.session_state["horizon"] = horizon
    if station:
        app.session_state["station"] = station
    return app.run()


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("horizon", ["15min", "1h", "dayahead"])
def test_page_runs_without_error(page, horizon):
    app = run_app(page, horizon)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_holdout_station_is_marked():
    app = run_app("Forecast vs actual", "1h", station="SQ3")
    assert not app.exception
    assert any("never trained" in info.value for info in app.info)


def test_injected_fault_is_reported():
    app = run_app("Fault flags", "1h")
    app.checkbox[0].check().run()
    assert not app.exception, [e.value for e in app.exception]
    messages = [m.value for m in app.success] + [m.value for m in app.error]
    assert any("Injected fault" in m for m in messages)
