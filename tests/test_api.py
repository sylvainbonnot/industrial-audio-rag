from fastapi.testclient import TestClient
from rag_audio.api import app
import socket
import pytest


def is_qdrant_running(host="localhost", port=6333):
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) == 0


@pytest.mark.skipif(not is_qdrant_running(), reason="Qdrant not running")
def test_ask_endpoint():
    client = TestClient(app)
    response = client.get("/ask", params={"q": "loud valve at 50 Hz"})
    assert response.status_code == 200
    assert "answer" in response.json()
