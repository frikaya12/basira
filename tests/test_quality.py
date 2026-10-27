import numpy as np
import pytest

from basira.preprocessing import crop_to_field, preprocess
from basira.quality import assess_quality, decode_image
from basira.synthetic import encode_png, make_fundus


@pytest.mark.parametrize("variant", ["saine", "lesions"])
@pytest.mark.parametrize("side", [1, -1])
def test_photo_nette_acceptee(variant, side):
    report = assess_quality(make_fundus(variant, side, seed=3))
    assert report.ok, report.reasons
    assert report.sharpness > 15


def test_photo_floue_refusee():
    report = assess_quality(make_fundus("floue", 1, seed=3))
    assert not report.ok
    assert "Photo floue" in report.reasons


def test_image_noire_refusee():
    report = assess_quality(np.zeros((300, 300, 3), dtype=np.uint8))
    assert not report.ok
    assert report.reasons == ["Image trop sombre ou vide"]


def test_image_surexposee_refusee():
    report = assess_quality(np.full((300, 300, 3), 250, dtype=np.uint8))
    assert "Photo trop claire" in report.reasons


def test_decode_png_aller_retour():
    img = make_fundus("saine", 1, 0)
    assert decode_image(encode_png(img)).shape == img.shape


def test_decode_refuse_un_fichier_non_image():
    with pytest.raises(ValueError):
        decode_image(b"ceci n'est pas une image")


def test_preprocess_forme_et_type():
    out = preprocess(make_fundus("lesions", 1, 0), size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_recadrage_supprime_le_fond_noir():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[100:300, 250:350] = 120
    assert crop_to_field(img).shape[:2] == (200, 100)
