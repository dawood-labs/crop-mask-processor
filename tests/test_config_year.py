"""Year templating: one config file serves the 2014-2026 seasons."""

import pytest

from cropmask.config import Config


# ------------------------------------------------------------ year templating
def test_year_fills_every_templated_uri(tmp_path):
    cfg = Config.load(
        None,
        year=2019,
        input_uri="gs://b/fao/{year}",
        output_uri="gs://b/fao/out/{year}",
        boundary_uri="gs://b/fao/boundary/x.shp",
        work_dir=str(tmp_path / "{year}"),
        report_name="acreage_{year}.xlsx",
    )
    assert cfg.input_uri == "gs://b/fao/2019"
    assert cfg.output_uri == "gs://b/fao/out/2019"
    assert cfg.report_name == "acreage_2019.xlsx"
    assert cfg.work_dir.endswith("2019")


def test_missing_year_is_an_error_not_a_literal_path():
    """A '{year}' left unsubstituted would create a directory called {year}."""
    with pytest.raises(ValueError, match="year"):
        Config.load(
            None,
            input_uri="gs://b/fao/{year}",
            output_uri="gs://b/fao/out/{year}",
            boundary_uri="gs://b/fao/boundary/x.shp",
        )


def test_config_without_templates_does_not_need_a_year():
    cfg = Config.load(
        None,
        input_uri="gs://b/fao/2025",
        output_uri="gs://b/fao/out/2025",
        boundary_uri="gs://b/fao/boundary/x.shp",
    )
    assert cfg.year is None
    assert cfg.input_uri == "gs://b/fao/2025"


def test_each_season_gets_its_own_output_prefix():
    """Two seasons must never share an output prefix or a resume state dir."""
    uris = []
    for year in (2014, 2020, 2026):
        cfg = Config.load(
            None, year=year,
            input_uri="gs://b/fao/{year}",
            output_uri="gs://b/fao/out/{year}",
            boundary_uri="gs://b/fao/boundary/x.shp",
        )
        uris.append(cfg.output_uri)
    assert len(set(uris)) == 3
