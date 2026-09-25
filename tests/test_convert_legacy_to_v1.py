from __future__ import annotations

import adios2py
import numpy as np
import pytest
import xarray as xr

import pscpy
from pscpy.convert.legacy_to_v1 import (
    OUTPUT_VERSION,
    convert_file,
    legacy_component_names,
    main,
)

LEGACY_DIR = pscpy.sample_dir / "legacy"
V1_DIR = pscpy.sample_dir / "v1"
SAMPLE_FILES = [
    "continuity.000000001.bp",
    "gauss.000000001.bp",
    "pfd.000000001.bp",
    "pfd_moments.000000001.bp",
]

# old samples without length/corner attrs; values match the old decode tests
LENGTH_400 = [1, 12.8, 51.2]
CORNER_400 = [0, -6.4, -25.6]


def _write_legacy(filename, variables, attrs):
    with adios2py.File(filename, mode="w") as file, file.steps.next() as step:
        for name, value in attrs.items():
            file.attrs[name] = value
        for name, data in variables.items():
            step[name] = data


def _assert_matches_v1(converted, expected):
    """Like assert_identical, but coordinates only need to be close."""
    for crd in "xyz":
        np.testing.assert_allclose(converted[crd], expected[crd])
    converted = converted.assign_coords({crd: expected[crd] for crd in "xyz"})
    xr.testing.assert_identical(converted, expected)


@pytest.fixture
def legacy_attrs():
    return {
        "time": 1.5,
        "step": np.int32(3),
        "length": np.array([1.0, 2.0, 3.0]),
        "corner": np.array([0.0, -1.0, 0.0]),
    }


@pytest.mark.parametrize("filename", SAMPLE_FILES)
def test_matches_psc_v1(tmp_path, filename):
    """Converting legacy output gives what the v1 writer produces."""
    dst = tmp_path / filename
    convert_file(LEGACY_DIR / filename, dst, species_names=["e", "i"])

    converted = xr.open_dataset(dst)
    expected = xr.open_dataset(V1_DIR / filename)
    _assert_matches_v1(converted, expected)
    for name, var in expected.variables.items():
        assert converted[name].dtype == var.dtype
        for key, value in var.attrs.items():
            assert np.array_equal(converted[name].attrs[key], value)


def test_moments_values(tmp_path):
    src = LEGACY_DIR / "pfd_moments.000000001.bp"
    dst = tmp_path / "pfd_moments.000000001.bp"
    convert_file(src, dst, species_names=["e", "i"])

    legacy = xr.open_dataset(src)
    converted = xr.open_dataset(dst)
    names = legacy_component_names("all_1st_cc", 26, ["e", "i"])
    for idx, name in enumerate(names):
        expected = legacy.all_1st_cc.isel(dim_0_1=0, dim_1_26=idx).data
        assert np.array_equal(converted[name].isel(time=0).data, expected)
    assert np.all(converted.rho_e < 0)
    assert np.all(converted.rho_i > 0)


@pytest.mark.parametrize(
    ("filename", "field"),
    [("pfd.000000400.bp", "jeh"), ("pfd_moments.000000400.bp", "all_1st")],
)
def test_values_and_dtype_preserved(tmp_path, filename, field):
    src = pscpy.sample_dir / filename
    dst = tmp_path / filename
    convert_file(
        src, dst, species_names=["e", "i"], length=LENGTH_400, corner=CORNER_400
    )

    legacy = xr.open_dataset(src)[field]
    converted = xr.open_dataset(dst)
    names = legacy_component_names(field, legacy.shape[1], ["e", "i"])
    assert set(converted.data_vars) == set(names)
    for idx, name in enumerate(names):
        assert converted[name].dtype == np.float32
        assert np.array_equal(converted[name].isel(time=0).data, legacy[0, idx].data)


def test_decode_converted(tmp_path):
    dst = tmp_path / "pfd.000000400.bp"
    convert_file(
        pscpy.sample_dir / "pfd.000000400.bp",
        dst,
        length=LENGTH_400,
        corner=CORNER_400,
    )

    ds = pscpy.decode_psc(xr.open_dataset(dst))
    assert ds.attrs["psc_output_version"] == OUTPUT_VERSION == "1.0.0"
    assert ds.jx_ec.sizes == dict(x=1, y=128, z=512)  # noqa: C408
    assert np.isclose(ds.time, 109.381, atol=1e-3)
    assert ds.attrs["step"] == 400
    dz = 51.2 / 512
    assert np.allclose(ds.z, np.linspace(-25.6, 25.6, 512, endpoint=False) + dz / 2)
    assert np.allclose(ds.x, [0.5])


def test_ib_im_preserved(tmp_path):
    dst = tmp_path / "pfd.000000400.bp"
    convert_file(
        pscpy.sample_dir / "pfd.000000400.bp", dst, length=LENGTH_400, corner=CORNER_400
    )
    ds = xr.open_dataset(dst)
    assert np.array_equal(ds.hz_fc.attrs["ib"], [0, 0, 0])
    assert np.array_equal(ds.hz_fc.attrs["im"], [1, 128, 128])


def test_time_as_array(tmp_path):
    """Some legacy files store time and step as 1-element arrays."""
    dst = tmp_path / "pfd.000000000.bp"
    convert_file(pscpy.sample_dir / "pfd.000000000.bp", dst)

    ds = xr.open_dataset(dst)
    assert ds.time.shape == (1,)
    assert ds.time[0] == 0.0
    assert np.ndim(ds.attrs["time"]) == 0
    assert np.ndim(ds.attrs["step"]) == 0


def test_override_corner_and_length(tmp_path, legacy_attrs):
    src = tmp_path / "legacy.bp"
    dst = tmp_path / "converted.bp"
    _write_legacy(src, {"dive": np.zeros((1, 2, 3, 4))}, legacy_attrs)
    convert_file(src, dst, length=[4.0, 6.0, 8.0], corner=[1.0, 2.0, 3.0])

    ds = xr.open_dataset(dst)
    assert np.array_equal(ds.attrs["length"], [4.0, 6.0, 8.0])
    assert np.array_equal(ds.attrs["corner"], [1.0, 2.0, 3.0])
    assert np.allclose(ds.x, [1.5, 2.5, 3.5, 4.5])
    assert np.allclose(ds.y, [3.0, 5.0, 7.0])
    assert np.allclose(ds.z, [5.0, 9.0])


@pytest.mark.parametrize("field", ["dive", "rho", "d_rho", "dt_divj"])
def test_single_component_fields(tmp_path, legacy_attrs, field):
    src = tmp_path / "legacy.bp"
    dst = tmp_path / "converted.bp"
    data = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    _write_legacy(src, {field: data}, legacy_attrs)
    convert_file(src, dst)

    ds = pscpy.decode_psc(xr.open_dataset(dst))
    assert set(ds.data_vars) == {field}
    assert ds[field].dims == ("z", "y", "x")
    assert np.array_equal(ds[field].data, data[0])
    assert ds.time == 1.5
    assert ds.attrs["step"] == 3


@pytest.mark.parametrize("missing", ["length", "corner"])
def test_missing_domain_attr(tmp_path, legacy_attrs, missing):
    src = tmp_path / "legacy.bp"
    del legacy_attrs[missing]
    _write_legacy(src, {"dive": np.zeros((1, 2, 3, 4))}, legacy_attrs)
    with pytest.raises(ValueError, match=missing):
        convert_file(src, tmp_path / "converted.bp")


def test_unknown_field(tmp_path, legacy_attrs):
    src = tmp_path / "legacy.bp"
    _write_legacy(src, {"foo": np.zeros((2, 2, 3, 4))}, legacy_attrs)
    with pytest.raises(ValueError, match="'foo'"):
        convert_file(src, tmp_path / "converted.bp")


def test_wrong_number_of_components(tmp_path, legacy_attrs):
    src = tmp_path / "legacy.bp"
    _write_legacy(src, {"jeh": np.zeros((8, 2, 3, 4))}, legacy_attrs)
    with pytest.raises(ValueError, match="8 components, expected 9"):
        convert_file(src, tmp_path / "converted.bp")


def test_moments_need_species(tmp_path):
    with pytest.raises(ValueError, match="species_names"):
        convert_file(LEGACY_DIR / "pfd_moments.000000001.bp", tmp_path / "converted.bp")


def test_moments_wrong_species_count(tmp_path):
    with pytest.raises(ValueError, match="26 components, expected 39"):
        convert_file(
            LEGACY_DIR / "pfd_moments.000000001.bp",
            tmp_path / "converted.bp",
            species_names=["e", "i", "p"],
        )


def test_already_converted(tmp_path):
    with pytest.raises(ValueError, match="not in the legacy format"):
        convert_file(V1_DIR / "pfd.000000001.bp", tmp_path / "converted.bp")


def test_dst_exists(tmp_path):
    dst = tmp_path / "converted.bp"
    dst.mkdir()
    with pytest.raises(FileExistsError):
        convert_file(LEGACY_DIR / "pfd.000000001.bp", dst)


def test_main(tmp_path, capsys):
    outdir = tmp_path / "out"
    main(
        [
            "-o",
            str(outdir),
            "--species",
            "e,i",
            *(str(LEGACY_DIR / name) for name in SAMPLE_FILES),
        ]
    )

    assert sorted(p.name for p in outdir.iterdir()) == SAMPLE_FILES
    for name in SAMPLE_FILES:
        _assert_matches_v1(
            xr.open_dataset(outdir / name), xr.open_dataset(V1_DIR / name)
        )
    assert "pfd_moments.000000001.bp" in capsys.readouterr().out


def test_main_refuses_to_overwrite():
    src = LEGACY_DIR / "pfd.000000001.bp"
    with pytest.raises(FileExistsError):
        main(["-o", str(LEGACY_DIR), str(src)])
