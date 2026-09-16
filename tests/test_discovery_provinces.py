"""A district filed under two province folders must not become two districts.

2023 had a byte-for-byte copy of SANGHAR's rice under Punjab; 2024 had a second,
different version of MIANWALI's rice under Sindh. Each was processed as a
separate district, published without being de-overlapped against that district's
other crops, and counted twice in the report.
"""

from cropmask.discovery import build_index

ROOT = "fao/crop_processing/2024"
KNOWN = {"mianwali", "sanghar", "lahore"}
BOUNDARY = {"mianwali": "punjab", "sanghar": "sindh", "lahore": "punjab"}


def shapefile(path, size=1_000):
    stem = f"{ROOT}/{path}"
    return [(stem + ext, size) for ext in (".shp", ".shx", ".dbf", ".prj")]


def listing(*files):
    return [item for f in files for item in f]


def mianwali_2024():
    return listing(
        shapefile("Cotton/Punjab/MIANWALI/Mianwali_Cotton"),
        shapefile("Fall Maize/Punjab/MIANWALI/MIANWALI_Emptyshapefile", size=100),
        shapefile("Rice/Punjab/MIANWALI/Mianwali_2024_Rice", size=2_150_000),
        shapefile("Sugarcane/Punjab/MIANWALI/Mianwali_cane"),
        # the stray: a different, larger version under the wrong province
        shapefile("Rice/Sindh/MIANWALI/MIANWALI", size=2_520_000),
    )


def test_stray_province_folder_does_not_create_a_second_district():
    tasks, anomalies = build_index(mianwali_2024(), ROOT, KNOWN, BOUNDARY)
    assert list(tasks) == [("punjab", "mianwali")]
    assert set(tasks[("punjab", "mianwali")].crops) == {"Cotton", "Fall Maize", "Rice", "Sugarcane"}


def test_the_correctly_filed_version_is_used_not_the_larger_stray():
    """Merging the stray in as a candidate would pick it for being bigger."""
    tasks, _ = build_index(mianwali_2024(), ROOT, KNOWN, BOUNDARY)
    rice = tasks[("punjab", "mianwali")].crops["Rice"]
    assert rice.shp_blob.endswith("Rice/Punjab/MIANWALI/Mianwali_2024_Rice.shp")


def test_the_ignored_file_is_reported():
    _, anomalies = build_index(mianwali_2024(), ROOT, KNOWN, BOUNDARY)
    stray = [a for a in anomalies if "wrong province folder" in a["issue"]]
    assert len(stray) == 1
    assert stray[0]["ignored"].endswith("Rice/Sindh/MIANWALI/MIANWALI.shp")
    assert stray[0]["district_province"] == "punjab"


def test_boundary_province_wins_over_majority():
    """SANGHAR shape, but with the stray folder holding more crops."""
    blobs = listing(
        shapefile("Rice/Sindh/SANGHAR/SANGHAR"),
        shapefile("Rice/Punjab/SANGHAR/SANGHAR"),
        shapefile("Cotton/Punjab/SANGHAR/SANGHAR_cotton"),
    )
    tasks, _ = build_index(blobs, ROOT, KNOWN, BOUNDARY)
    assert list(tasks) == [("sindh", "sanghar")]


def test_without_boundary_provinces_the_majority_folder_wins():
    tasks, anomalies = build_index(mianwali_2024(), ROOT, KNOWN, None)
    assert list(tasks) == [("punjab", "mianwali")]
    assert any("wrong province folder" in a["issue"] for a in anomalies)


def test_an_unresolvable_tie_skips_the_district_loudly():
    blobs = listing(
        shapefile("Rice/Punjab/MIANWALI/a"),
        shapefile("Cotton/Sindh/MIANWALI/b"),
    )
    tasks, anomalies = build_index(blobs, ROOT, KNOWN, None)
    assert tasks == {}
    assert any("no way to tell" in a["issue"] for a in anomalies)


def test_a_single_province_folder_is_never_second_guessed():
    """Only conflicts are resolved; an unusual folder name alone is fine."""
    blobs = listing(shapefile("Rice/Pb/LAHORE/LAHORE"), shapefile("Cotton/Pb/LAHORE/LAHORE_c"))
    tasks, anomalies = build_index(blobs, ROOT, KNOWN, BOUNDARY)
    assert list(tasks) == [("pb", "lahore")]
    assert not any("province" in a["issue"] for a in anomalies)
