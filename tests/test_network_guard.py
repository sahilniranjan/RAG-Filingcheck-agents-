"""Tests for the pytest network guard in conftest.py."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest


def _echo_once(server: socket.socket) -> None:
    """Accept one connection, echo what it receives, then close."""
    conn, _ = server.accept()
    with conn:
        data = conn.recv(1024)
        conn.sendall(b"echo:" + data)


def _start_echo_server(server: socket.socket) -> threading.Thread:
    """Run _echo_once on a background thread for an already listening socket."""
    thread = threading.Thread(target=_echo_once, args=(server,), daemon=True)
    thread.start()
    return thread


@pytest.fixture
def tcp_server() -> Iterator[socket.socket]:
    """A TCP server listening on 127.0.0.1 with an OS-chosen port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)
    yield server
    server.close()


def test_guard_is_installed(network_guard: SimpleNamespace) -> None:
    assert socket.socket.connect is not network_guard.originals.connect
    assert socket.getaddrinfo is not network_guard.originals.getaddrinfo


def test_create_connection_to_external_host_is_blocked(network_guard: SimpleNamespace) -> None:
    with pytest.raises(network_guard.error, match="example.com"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_raw_connect_to_public_ip_is_blocked(network_guard: SimpleNamespace) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(network_guard.error, match="93.184.215.14"):
            sock.connect(("93.184.215.14", 80))
        with pytest.raises(network_guard.error):
            sock.connect_ex(("93.184.215.14", 80))


def test_error_message_explains_what_to_do(network_guard: SimpleNamespace) -> None:
    with pytest.raises(network_guard.error) as excinfo:
        socket.getaddrinfo("example.com", 443)
    message = str(excinfo.value)
    assert "must not use the network" in message
    assert "TIKTOKEN_CACHE_DIR" in message
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_localhost_tcp_is_allowed(tcp_server: socket.socket, host: str) -> None:
    port = tcp_server.getsockname()[1]
    thread = _start_echo_server(tcp_server)
    with socket.create_connection((host, port), timeout=5) as client:
        client.sendall(b"ping")
        assert client.recv(1024) == b"echo:ping"
    thread.join(timeout=5)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no unix sockets on this platform")
def test_unix_socket_is_allowed() -> None:
    # macOS caps unix socket paths near 104 bytes, so use a short dir under /tmp.
    short_dir = tempfile.mkdtemp(prefix="fc", dir="/tmp")
    path = os.path.join(short_dir, "s.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(path)
        server.listen(1)
        server.settimeout(5)
        thread = _start_echo_server(server)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(path)
            client.sendall(b"hello")
            assert client.recv(1024) == b"echo:hello"
        thread.join(timeout=5)
    finally:
        server.close()
        shutil.rmtree(short_dir, ignore_errors=True)


def test_getaddrinfo_allows_localhost_and_blocks_external(network_guard: SimpleNamespace) -> None:
    assert socket.getaddrinfo("localhost", 80)
    assert socket.getaddrinfo(None, 80)
    with pytest.raises(network_guard.error, match="example.com"):
        socket.getaddrinfo("example.com", 80)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("127.5.6.7", True),
        ("::1", True),
        (None, True),
        ("", True),
        ("example.com", False),
        ("93.184.215.14", False),
        ("10.0.0.1", False),
    ],
)
def test_is_local_host(network_guard: SimpleNamespace, host: str | None, expected: bool) -> None:
    assert network_guard.is_local_host(host) is expected


# ---------------------------------------------------------------------------
# DATABASE_URL must point at a local database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("postgresql://filingcheck:filingcheck@localhost:5432/filingcheck", True),
        ("postgresql://u:p@LOCALHOST/x", True),
        ("postgresql://u:p@127.0.0.1:5432/x", True),
        ("postgresql://u:p@[::1]:5432/x", True),
        ("postgresql:///filingcheck", True),
        ("postgresql://user@/db?host=/var/run/postgresql", True),
        ("postgresql://u:p@db.example.com:5432/x", False),
        ("postgresql://u:p@10.0.0.5/x", False),
        ("postgresql://localhost@db.example.com/x", False),
        ("postgresql://localhost:pw@db.example.com/x", False),
        ("postgresql://u:p@localhost.example.com/x", False),
    ],
)
def test_database_url_is_local(
    network_guard: SimpleNamespace, url: str | None, expected: bool
) -> None:
    assert network_guard.database_url_is_local(url) is expected


def test_remote_database_url_stops_the_session(tmp_path: Path) -> None:
    conftest = Path(__file__).with_name("conftest.py")
    shutil.copy(conftest, tmp_path / "conftest.py")
    (tmp_path / "test_never_runs.py").write_text("def test_x():\n    assert False\n")
    env = {**os.environ, "DATABASE_URL": "postgresql://u:secret@db.example.com/x"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Refusing to run tests" in output
    assert "db.example.com" in output
    assert "secret" not in output
    assert "test_never_runs" not in output
