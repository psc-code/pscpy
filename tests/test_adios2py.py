from __future__ import annotations

import adios2
import numpy as np
import pytest

import pscpy
from pscpy import adios2py


@pytest.fixture
def pfd_file():
    return adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")


@pytest.fixture
def test_file(tmp_path):
    filename = tmp_path / "test_file.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

    return adios2py.File(filename, mode="r")


def test_open_close(pfd_file):
    pfd_file.close()


def test_open_twice():
    file1 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841
    file2 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841


def test_with(pfd_file):
    with pfd_file:
        pass


def test_file_repr(pfd_file):
    print(repr(pfd_file))
    assert repr(pfd_file).startswith(f"{type(pfd_file)}(filename=")


def test_variable_names(pfd_file):
    assert pfd_file.variable_names == set({"jeh"})
    assert pfd_file.attribute_names == set({"ib", "im", "step", "time"})


def test_get_variable(pfd_file):
    var = pfd_file.get_variable("jeh")
    assert var.name == "jeh"
    assert var.shape == (1, 128, 512, 9)
    assert var.dtype == np.float32


def test_get_variable_not_found(pfd_file):
    with pytest.raises(ValueError, match="not found"):
        pfd_file.get_variable("xyz")


def test_variable_bool(pfd_file):
    with pfd_file:
        var = pfd_file.get_variable("jeh")
        assert var
        assert var.shape == (1, 128, 512, 9)

    assert not var


def test_variable_shape(pfd_file):
    with pfd_file:
        var = pfd_file.get_variable("jeh")
        assert var.shape == (1, 128, 512, 9)

    with pytest.raises(ValueError, match="variable is closed"):
        assert var.shape == (1, 128, 512, 9)


def test_variable_name(pfd_file):
    with pfd_file:
        var = pfd_file.get_variable("jeh")
        assert var.name == "jeh"

    with pytest.raises(ValueError, match="variable is closed"):
        assert var.name == "jeh"


def test_variable_dtype(pfd_file):
    with pfd_file:
        var = pfd_file.get_variable("jeh")
        assert var.dtype == np.float32

    with pytest.raises(ValueError, match="variable is closed"):
        assert var.dtype == np.float32


def test_variable_repr(pfd_file):
    with pfd_file:
        var = pfd_file.get_variable("jeh")
        assert (
            repr(var) == f"{type(var)}(name=jeh, shape=(1, 128, 512, 9), dtype=float32"
        )

    assert repr(var) == f"{type(var)} (closed)"


def test_variable_is_reverse_dims(pfd_file):
    var = pfd_file.get_variable("jeh")
    assert not var.is_reverse_dims

    # with adios2py.File(
    #     "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
    # ) as file:
    #     var = file.get_variable("pot")
    #     assert var.is_reverse_dims


def test_variable_getitem_scalar(test_file):
    test_file.begin_step()
    var = test_file.get_variable("scalar")
    assert var[()] == 0
    test_file.end_step()


def test_variable_getitem_arr1d(test_file):
    test_file.begin_step()
    var = test_file.get_variable("arr1d")
    assert np.all(var[()] == np.arange(10))
    test_file.end_step()


def test_variable_getitem_arr1d_indexing(test_file):
    test_file.begin_step()
    var = test_file.get_variable("arr1d")
    assert var[2] == 2
    assert np.all(var[2:4] == np.arange(10)[2:4])
    assert np.all(var[:] == np.arange(10)[:])
    test_file.end_step()


@pytest.mark.xfail
def test_variable_getitem_arr1d_indexing_step(test_file):
    test_file.begin_step()
    var = test_file.get_variable("arr1d")
    assert np.all(var[::2] == np.arange(10)[::2])
    test_file.end_step()


@pytest.mark.xfail
def test_variable_getitem_arr1d_indexing_reverse(test_file):
    test_file.begin_step()
    var = test_file.get_variable("arr1d")
    assert np.all(var[::-1] == np.arange(10)[::-1])
    test_file.end_step()


def test_variable_array(test_file):
    test_file.begin_step()
    scalar = np.asarray(test_file.get_variable("scalar"))
    assert np.array_equal(scalar, 0)
    arr1d = np.asarray(test_file.get_variable("arr1d"))
    assert np.array_equal(arr1d, np.arange(10))
    test_file.end_step()


def test_get_attribute(pfd_file):
    assert all(pfd_file.get_attribute("ib") == (0, 0, 0))
    assert all(pfd_file.get_attribute("im") == (1, 128, 128))
    assert np.isclose(pfd_file.get_attribute("time"), 109.38)
    assert pfd_file.get_attribute("step") == 400


def test_write_streaming(tmp_path):
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="w") as file:
        print("file", file)
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)


def test_read_streaming_adios2(tmp_path):
    test_write_streaming(tmp_path)  # type: ignore[no-untyped-call]
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="r") as file:
        for n, step in enumerate(file):
            scalar = step.read("scalar")
            assert scalar == n
        assert n == 4


def test_read_streaming_adios2_step_persist(tmp_path):
    test_write_streaming(tmp_path)  # type: ignore[no-untyped-call]
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="r") as file:
        for n, step in enumerate(file):
            if n == 1:
                step1 = step

        # This may be confusing, but behaves as designed
        assert step1.read("scalar") == 4


def test_read_streaming_adios2py(test_file):
    for n, step in enumerate(test_file.steps()):
        scalar = step.get_variable("scalar")[()]
        assert scalar == n
    assert n == 4


def test_read_streaming_adios2py_mixed(test_file):
    test_file.begin_step()
    assert test_file.get_variable("scalar")[()] == 0
    test_file.end_step()

    for step in test_file.steps():
        scalar = test_file.get_variable("scalar")[()]
        assert scalar == step.current_step()
    assert test_file.current_step() == 4


@pytest.mark.xfail
def test_read_streaming_adios2py_step_persist(test_file):
    for n, step in enumerate(test_file.steps()):
        if n == 1:
            step1 = step

    assert step1.get_variable("scalar")[()] == 1


@pytest.mark.xfail
def test_read_streaming_adios2py_current_step_0(test_file):
    assert test_file.current_step() is None


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
