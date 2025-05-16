from rag_audio.indexer import parse_filename
from pathlib import Path


# def test_valid_filename():
#     name = Path("bearing_01_source_train_normal_000123.wav")
#     meta = parse_filename(name)
#     f = Path("bearing_01_source_train_normal_000123.wav")
#     meta = parse_filename(f)
#     print(meta)
#     assert meta["machine_type"] == "bearing"
#     assert meta["machine_type"] == "bearing"
#     assert meta["clip_id"] == "000123"


def test_valid_filename(tmp_path):
    subdir = tmp_path / "bearing"
    subdir.mkdir()
    wav = subdir / "bearing_01_source_train_normal_000123.wav"
    wav.write_bytes(b"RIFF....")  # fake or real WAV header
    meta = parse_filename(wav)
    assert meta["machine_type"] == "bearing"
