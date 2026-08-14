import pytest

from server.geometry.step_io import (
    StepError,
    face_map,
    get_face,
    load_step,
    save_step,
    shape_volume,
)


def test_load_box(box_step):
    s = load_step(box_step)
    assert face_map(s).Size() == 6
    assert abs(shape_volume(s) - 60 * 40 * 8) < 1e-6


def test_roundtrip(box_step, tmp_path):
    s = load_step(box_step)
    out = tmp_path / "rt.step"
    save_step(s, out)
    s2 = load_step(out)
    assert abs(shape_volume(s2) - shape_volume(s)) < 1e-3


def test_get_face(box_step):
    s = load_step(box_step)
    f = get_face(s, 1)
    assert not f.IsNull()
    with pytest.raises(StepError):
        get_face(s, 99)


def test_load_missing():
    with pytest.raises(StepError):
        load_step("nope.step")
