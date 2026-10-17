import pytest

from receipt_reader import read_receipt_impl


@pytest.fixture()
def sample_image(tmp_path):
    from PIL import Image, ImageDraw
    path = str(tmp_path / "challan.jpg")
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "LOADING CHALLAN", fill="black")
    d.text((20, 60), "Sender: Sathya Traders Chennai", fill="black")
    d.text((20, 100), "Pallets: 6", fill="black")
    d.text((20, 140), "Weight: 1500 kg", fill="black")
    d.text((20, 180), "Destination: Sriperumbudur", fill="black")
    img.save(path, "JPEG")
    return path


def test_rejects_missing_file(tmp_path):
    res = read_receipt_impl(str(tmp_path / "nope.jpg"))
    assert res["status"] == "ERROR"
    assert res["reason"] == "not_found"


def test_gemini_mocked_extraction(sample_image, monkeypatch):
    """Without network: patch the genai client response (deterministic unit test)."""
    class FakeResp:
        text = ('Sender: Sathya Traders Chennai; Pallets: 6;'
                ' Weight: 1500 kg; Destination: Sriperumbudur')

    class FakeModels:
        def generate_content(self, *a, **kw):
            return FakeResp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr("receipt_reader._client", lambda: FakeClient())
    res = read_receipt_impl(sample_image)
    assert res["status"] == "OK"
    assert res["pallets"] == 6
    assert res["weight_kg"] == 1500
    assert "Sathya" in res["sender"]
