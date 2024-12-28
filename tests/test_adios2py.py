from __future__ import annotations

import adios2
import numpy as np
import pytest

import pscpy
from pscpy import adios2py


@pytest.fixture
def pfd_file():
    return adios2py.File(pscpy.sample_dir / "pfd.000000400.bp", mode="rra")


@pytest.fixture
def test_filename(tmp_path):
    filename = tmp_path / "test_file.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

    return filename


@pytest.fixture
def test_file(test_filename):
    return adios2py.File(test_filename, mode="r")


def test_open_close(pfd_file):
    assert pfd_file  # is open
    pfd_file.close()
    assert not pfd_file  # is closed


def test_open_twice():
    file1 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841
    file2 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841


def test_open_with_parameters():
    params = {"OpenTimeoutSecs": "15"}
    with adios2py.File(
        pscpy.sample_dir / "pfd.000000400.bp", parameters=params
    ) as file:
        assert file._state.io.parameters() == params


def test_open_with_engine():
    with adios2py.File(
        pscpy.sample_dir / "pfd.000000400.bp", engine_type="BP4"
    ) as file:
        assert file._state.io.engine_type() == "BP4"


def test_with(pfd_file):
    with pfd_file:
        pass


def test_file_repr(pfd_file):
    assert repr(pfd_file)


def test_keys(pfd_file):
    assert pfd_file.keys() == set({"jeh"})


def test_attrs_keys(pfd_file):
    assert pfd_file.attrs.keys() == set({"ib", "im", "step", "time"})


def test_attrs_contains(pfd_file):
    assert "ib" in pfd_file.attrs
    assert "ix" not in pfd_file.attrs


def test_attrs_iter(pfd_file):
    assert set(pfd_file.attrs) == set({"ib", "im", "step", "time"})


def test_get_variable(pfd_file):
    var = pfd_file["jeh"]
    assert var.name == "jeh"
    assert var.shape == (9, 512, 128, 1)
    assert var.dtype == np.float32


def test_get_variable_not_found(pfd_file):
    with pytest.raises(KeyError):
        pfd_file["xyz"]


def test_variable_bool(pfd_file):
    with pfd_file:
        var = pfd_file["jeh"]
        assert var
        assert var.shape == (9, 512, 128, 1)

    assert not var


def test_variable_shape(pfd_file):
    with pfd_file:
        var = pfd_file["jeh"]
        assert var.shape == (9, 512, 128, 1)

    with pytest.raises(ValueError, match="is closed"):
        assert var.shape == (9, 512, 128, 1)


def test_variable_name(pfd_file):
    with pfd_file:
        var = pfd_file["jeh"]
        assert var.name == "jeh"

    with pytest.raises(ValueError, match="is closed"):
        assert var.name == "jeh"


def test_variable_dtype(pfd_file):
    with pfd_file:
        var = pfd_file["jeh"]
        assert var.dtype == np.float32

    with pytest.raises(ValueError, match="is closed"):
        assert var.dtype == np.float32


def test_variable_repr(pfd_file):
    with pfd_file:
        var = pfd_file["jeh"]
        assert (
            repr(var)
            == "adios2py.Variable(name=jeh, shape=(9, 512, 128, 1), dtype=float32"
        )

    assert repr(var) == "adios2py.Variable(name=jeh) (closed)"


def test_variable_is_reverse_dims(pfd_file):
    var = pfd_file["jeh"]
    assert not var._is_reverse_dims()

    # with adios2py.File(
    #     "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
    # ) as file:
    #     var = file["pot"]
    #     assert var.is_reverse_dims


def test_variable_getitem_scalar(test_file):
    for step in test_file.steps:
        var = step["scalar"]
        assert var[()] == step.step()


def test_variable_getitem_arr1d(test_file):
    for step in test_file.steps:
        var = step["arr1d"]
        assert np.all(var[()] == np.arange(10))


def test_variable_getitem_arr1d_indexing(test_file):
    for step in test_file.steps:
        var = step["arr1d"]
        assert var[2] == 2
        assert np.all(var[2:4] == np.arange(10)[2:4])
        assert np.all(var[:] == np.arange(10)[:])


@pytest.mark.xfail
def test_variable_getitem_arr1d_indexing_step(test_file):
    for step in test_file.steps:
        var = step["arr1d"]
        assert np.all(var[::2] == np.arange(10)[::2])


@pytest.mark.xfail
def test_variable_getitem_arr1d_indexing_reverse(test_file):
    for step in test_file.steps:
        var = step["arr1d"]
        assert np.all(var[::-1] == np.arange(10)[::-1])


def test_variable_array(test_file):
    for step in test_file.steps:
        scalar = np.asarray(step["scalar"])
        assert np.array_equal(scalar, step.step())
        arr1d = np.asarray(step["arr1d"])
        assert np.array_equal(arr1d, np.arange(10))


def test_get_attribute(pfd_file):
    assert all(pfd_file.attrs["ib"] == (0, 0, 0))
    assert all(pfd_file.attrs["im"] == (1, 128, 128))
    assert np.isclose(pfd_file.attrs["time"], 109.38)
    assert pfd_file.attrs["step"] == 400


def test_write_streaming(tmp_path):
    with adios2.Stream(str(tmp_path / "test_streaming.bp"), mode="w") as file:
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


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_read_adios2py(test_filename, mode):
    with adios2py.File(test_filename, mode=mode) as file:
        for n, step in enumerate(file.steps):
            scalar = step["scalar"][()]
            assert scalar == n
        assert n == 4


def test_read_streaming_adios2py_resume(test_file):
    # do 0th iteration
    for step in test_file.steps:
        scalar = step["scalar"][()]
        assert step.step() == 0
        assert scalar == 0
        break

    # then do the rest separately
    for n, step in enumerate(test_file.steps):
        scalar = step["scalar"][()]
        assert step.step() == n + 1
        assert scalar == n + 1
    assert n == 3


def test_read_streaming_adios2py_next(test_file):
    # do 0th iteration
    with next(test_file.steps) as step:
        scalar = step["scalar"][()]
        assert scalar == 0

    # then do the rest separately
    for n, step in enumerate(test_file.steps):
        scalar = step["scalar"][()]
        assert step.step() == n + 1
        assert scalar == n + 1
    assert n == 3


@pytest.mark.parametrize("mode", ["rra", "r"])
def test_read_adios2py_step_persist(test_filename, mode):
    with adios2py.File(test_filename, mode=mode) as file:
        for n, step in enumerate(file.steps):
            if n == 1:
                step1 = step

        assert step1.step() == 1
        if mode == "rra":
            assert step1["scalar"][()] == 1
        else:
            with pytest.raises(KeyError):
                step1["scalar"][()]


@pytest.mark.parametrize("mode", ["rra", "r"])
def test_read_adios2py_var_persist_r(test_filename, mode):
    with adios2py.File(test_filename, mode=mode) as file:
        for n, step in enumerate(file.steps):
            if n == 1:
                var1 = step["scalar"]

        if mode == "rra":
            assert var1[()] == 1
        else:
            with pytest.raises(KeyError, match="cannot access"):
                var1[()]


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_read_adios2py_steps_len(test_filename, mode):
    with adios2py.File(test_filename, mode=mode) as file:
        assert len(file.steps) == 5


def test_read_adios2py_steps_getitem_rra(test_filename):
    with adios2py.File(test_filename, mode="rra") as file:
        step = file.steps[2]
        assert step["scalar"][()] == 2


def test_read_adios2py_steps_getitem_r(test_filename):
    with adios2py.File(test_filename, mode="r") as file:  # noqa: SIM117
        with pytest.raises(TypeError):
            file.steps[2]


def test_read_streaming_adios2py_current_step_outside_step(test_file):
    assert test_file._state.current_step() is None


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_write_read_mixed(tmp_path, mode):
    filename = tmp_path / "mixed.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        file.write("scalar", -1)
        arr1d = np.arange(5)
        file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

        # these steps just get lost...
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

    with adios2.Stream(str(filename), mode=mode) as file:
        assert file.num_steps() == 1


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_write_read(tmp_path, mode):
    filename = tmp_path / "steps.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

    with adios2.Stream(str(filename), mode=mode) as file:
        assert file.num_steps() == 5


def test_read_steps(test_filename):
    with adios2py.File(test_filename, mode="rra") as file:
        assert len(file.steps) == 5
        var_scalar = file["scalar"]
        assert var_scalar.shape == (5,)
        # assert np.all(var_scalar[:] == np.arange(5))


# def test_single_value():
#     with adios2py.File(
#         "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
#     ) as file:
#         assert "dacttime" in file.variable_names
#         var = file["dacttime"]
#         val = var[()]
#         assert np.isclose(val, 1.4897556e09)
