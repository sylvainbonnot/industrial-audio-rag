from rag_audio.indexer import _process_file
from scipy.io.wavfile import write
import numpy as np


# def test_process_file_structure(tmp_path):
#     wav = tmp_path / "bearing_01_source_train_normal_000001.wav"
#     sr = 16000
#     t = np.linspace(0, 1, sr, False)
#     tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
#     write(wav, sr, tone)  # wav is the tmp_path destination
#     # wav.write_bytes(b"\x00\x00")  # dummy .wav file

#     point = _process_file(wav, embedder=lambda x: [0.1] * 1024)
#     assert "vector" in point
#     assert len(point.vector) == 1024
#     assert "payload" in point
#     assert "machine_type" in point.payload


def test_process_file_structure(tmp_path):
    from rag_audio.indexer import _process_file
    from scipy.io.wavfile import write
    import numpy as np

    # Create subfolder with machine type name
    subdir = tmp_path / "bearing"
    subdir.mkdir()

    wav = subdir / "bearing_01_source_train_normal_000001.wav"

    # Write valid .wav content
    sr = 16000
    t = np.linspace(0, 1, sr, False)
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    write(wav, sr, tone)

    # Dummy embedder
    class DummyEmbedder:
        def encode(self, text):
            return np.ones(1024, dtype=np.float32)

    point = _process_file(wav, embedder=DummyEmbedder())

    assert point.payload["machine_type"] == "bearing"
