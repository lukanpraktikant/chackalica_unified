"""Pure-numpy tests for the generic preprocess/postprocess (no torch/ORT needed)."""

import numpy as np
import pytest

from onnx_infer.meta import InputSpec, ModelMeta, Normalize
from onnx_infer.preprocess import Transform, _resize_chw, preprocess
from onnx_infer.postprocess import to_friendy


def _meta(**input_kwargs):
    return ModelMeta(
        arch="retinanet",
        num_classes=2,
        class_map={0: "a", 1: "b"},
        score_threshold=0.05,
        input=InputSpec(**input_kwargs),
        normalize=None,
    )


def test_resize_none_is_identity():
    img = np.random.rand(3, 40, 50).astype(np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="none"))
    assert batched.shape == (1, 3, 40, 50)
    np.testing.assert_array_equal(batched[0], img)
    assert (tf.scale_x, tf.scale_y, tf.pad_x, tf.pad_y) == (1.0, 1.0, 0, 0)
    assert (tf.orig_w, tf.orig_h) == (50, 40)


def test_pad_to_multiple_bottom_right():
    img = np.ones((3, 30, 50), dtype=np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="none", multiple=32, pad_value=0.0))
    assert batched.shape == (1, 3, 32, 64)  # ceil(30->32), ceil(50->64)
    assert np.all(batched[0, :, :30, :50] == 1.0)      # original preserved, top-left
    assert np.all(batched[0, :, 30:, :] == 0.0)         # bottom pad
    assert np.all(batched[0, :, :, 50:] == 0.0)         # right pad
    assert (tf.pad_x, tf.pad_y) == (0, 0)               # bottom-right pad => no origin shift


def test_normalize_applied():
    img = np.full((3, 8, 8), 0.5, dtype=np.float32)
    meta = ModelMeta(
        arch="rtdetr", num_classes=1, class_map={0: "a"}, score_threshold=0.5,
        input=InputSpec(resize_mode="none"),
        normalize=Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    )
    batched, _ = preprocess(img, meta)
    np.testing.assert_allclose(batched[0], 0.0, atol=1e-6)  # (0.5-0.5)/0.25 == 0


def test_byte_scale():
    img = np.full((3, 4, 4), 1.0, dtype=np.float32)
    batched, _ = preprocess(img, _meta(resize_mode="none", input_scale="byte"))
    np.testing.assert_allclose(batched[0], 255.0, atol=1e-4)


def test_resize_constant_preserved():
    img = np.full((3, 10, 12), 0.7, dtype=np.float32)
    out = _resize_chw(img, 20, 8)
    assert out.shape == (3, 20, 8)
    np.testing.assert_allclose(out, 0.7, atol=1e-6)  # bilinear of a constant is constant


def test_square_transform_scales():
    img = np.random.rand(3, 100, 200).astype(np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="square", size=50))
    assert batched.shape == (1, 3, 50, 50)
    assert tf.scale_x == 50 / 200
    assert tf.scale_y == 50 / 100


def test_to_friendy_maps_back_identity_transform():
    # xyxy [10,20,30,60] in a 100(w) x 200(h) image; scale=1, pad=0.
    boxes = np.array([[10, 20, 30, 60]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    tf = Transform(scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0, orig_w=100, orig_h=200)
    out = to_friendy(boxes, scores, labels, tf, score_threshold=0.05)
    # cx=20,cy=40,w=20,h=40 -> normalized by (100,200)
    np.testing.assert_allclose(out[0, :4], [0.2, 0.2, 0.2, 0.2], atol=1e-6)
    assert out[0, 4] == pytest.approx(0.9)
    assert out[0, 5] == 1.0


def test_to_friendy_inverts_scale_and_pad():
    # A box in a resized+padded frame maps back through the recorded transform.
    tf = Transform(scale_x=0.5, scale_y=0.5, pad_x=4, pad_y=6, orig_w=100, orig_h=100)
    # original box xyxy [20,20,40,40] -> input px: *0.5 then +pad = [14,16,24,26]
    boxes = np.array([[14, 16, 24, 26]], dtype=np.float32)
    out = to_friendy(boxes, np.array([0.8], np.float32), np.array([0], np.int64), tf, 0.05)
    # back to original: cx=30,cy=30,w=20,h=20 over 100 -> 0.3,0.3,0.2,0.2
    np.testing.assert_allclose(out[0, :4], [0.3, 0.3, 0.2, 0.2], atol=1e-5)


def test_to_friendy_threshold_and_empty():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.1], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    tf = Transform(1.0, 1.0, 0, 0, 100, 100)
    out = to_friendy(boxes, scores, labels, tf, score_threshold=0.5)
    assert out.shape == (1, 6)  # only the 0.9 box survives
    empty = to_friendy(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                       np.zeros((0,), np.int64), tf, 0.5)
    assert empty.shape == (0, 6)
