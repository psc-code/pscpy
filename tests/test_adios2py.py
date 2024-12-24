from __future__ import annotations

import adios2
import numpy as np
import pytest

import pscpy
from pscpy import adios2py


def test_open_close():
    file = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")
    file.close()


def test_open_twice():
    file1 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841
    file2 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841


def test_with():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp"):
        pass


def test_variable_names():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        assert file.variable_names == set({"jeh"})
        assert file.attribute_names == set({"ib", "im", "step", "time"})


def test_get_variable():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        var = file.get_variable("jeh")
        assert var.name == "jeh"
        assert var.shape == (1, 128, 512, 9)
        assert var.dtype == np.float32


def test_variable_closed():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        var = file.get_variable("jeh")
        assert var._shape() == (1, 128, 512, 9)

    with pytest.raises(ValueError, match="variable is closed"):
        assert var._shape() == (1, 128, 512, 9)


def test_is_reverse_dims():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        var = file.get_variable("jeh")
        assert var.name == "jeh"
        assert var.shape == (1, 128, 512, 9)
        assert var.dtype == np.float32
        assert not var.is_reverse_dims

    # with adios2py.File(
    #     "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
    # ) as file:
    #     var = file.get_variable("pot")
    #     assert var.is_reverse_dims


def test_get_attribute():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        assert all(file.get_attribute("ib") == (0, 0, 0))
        assert all(file.get_attribute("im") == (1, 128, 128))
        assert np.isclose(file.get_attribute("time"), 109.38)
        assert file.get_attribute("step") == 400


def test_write_streaming(tmp_path):
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="w") as file:
        print("file", file)
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)


def test_read_streaming_adios2(tmp_path):
    test_write_streaming(tmp_path)  # type: ignore[no-untyped-call]
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="r") as file:
        for step, _ in enumerate(file):
            scalar = file.read("scalar")
            assert step == scalar
        assert step == 4


def test_read_streaming_adios2py(tmp_path):
    test_write_streaming(tmp_path)  # type: ignore[no-untyped-call]
    with adios2py.File(tmp_path / "test_streaming.bp", mode="r") as file:
        for step in range(file.num_steps()):
            file.begin_step()
            scalar = file.get_variable("scalar")[()]
            assert step == scalar
            file.end_step()
        assert step == 4


# def test_single_value():
#     with adios2py.File(
#         "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
#     ) as file:
#         assert "dacttime" in file.variable_names
#         var = file.get_variable("dacttime")
#         val = var[()]
#         assert np.isclose(val, 1.4897556e09)


def test_construct_from_engine_io():
    ad = adios2.Adios()
    io = ad.declare_io("io_name")
    engine = io.open(
        str(pscpy.sample_dir / "pfd.000000400.bp"), adios2.bindings.Mode.Read
    )
    with adios2py.File((io, engine)) as file:
        assert all(file.get_attribute("ib") == (0, 0, 0))
        assert all(file.get_attribute("im") == (1, 128, 128))

    engine.close()
    ad.remove_io("io_name")
