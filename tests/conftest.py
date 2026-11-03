"""Shared pytest setup for FilingCheck.

This file installs a network guard before any test module is imported. Unit
tests must never touch the network, so any attempt to connect to a non-local
address, or to resolve a non-local hostname, raises NetworkBlockedError before
a single packet (including DNS) leaves the machine. Localhost and unix sockets
stay allowed so tests can run small local servers.

It also refuses to start the session at all if DATABASE_URL points at a non-local
database host, so a stray production or cloud URL in the environment can never be
used by a test.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}

DATABASE_URL_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
"""Hosts DATABASE_URL may name during tests. A missing host (unix socket) is also fine."""

# Keep the real functions so the guard can call through for local traffic and
# so its own tests can check what was patched.
ORIGINALS = SimpleNamespace(
    connect=socket.socket.connect,
    connect_ex=socket.socket.connect_ex,
    getaddrinfo=socket.getaddrinfo,
    gethostbyname=socket.gethostbyname,
    gethostbyname_ex=socket.gethostbyname_ex,
)


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach a non-local network address."""

    def __init__(self, host: object) -> None:
        super().__init__(
            f"Network access to {host!r} is blocked: unit tests must not use the network. "
            "Mock EDGAR and every LLM call. For tiktoken, set TIKTOKEN_CACHE_DIR to a "
            "pre-downloaded cache (CI pre-fetches it before pytest)."
        )
        self.host = host


def is_local_host(host: object) -> bool:
    """Return True if host is None, empty, a localhost name, or a loopback IP."""
    if host is None or host == "" or host == b"":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    name = host.strip().lower().rstrip(".")
    if name in LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check_address(sock: socket.socket, address: Any) -> None:
    """Raise NetworkBlockedError unless the socket address is local or AF_UNIX."""
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not is_local_host(host):
        raise NetworkBlockedError(host)


def _guarded_connect(self: socket.socket, address: Any) -> None:
    """socket.connect that refuses non-local addresses."""
    _check_address(self, address)
    return ORIGINALS.connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    """socket.connect_ex that refuses non-local addresses."""
    _check_address(self, address)
    return ORIGINALS.connect_ex(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    """getaddrinfo that refuses non-local names before any DNS lookup."""
    if not is_local_host(host):
        raise NetworkBlockedError(host)
    return ORIGINALS.getaddrinfo(host, *args, **kwargs)


def _guarded_gethostbyname(host: str) -> str:
    """gethostbyname that refuses non-local names before any DNS lookup."""
    if not is_local_host(host):
        raise NetworkBlockedError(host)
    return ORIGINALS.gethostbyname(host)


def _guarded_gethostbyname_ex(host: str) -> Any:
    """gethostbyname_ex that refuses non-local names before any DNS lookup."""
    if not is_local_host(host):
        raise NetworkBlockedError(host)
    return ORIGINALS.gethostbyname_ex(host)


def install_network_guard() -> None:
    """Patch the socket module so only local traffic is allowed."""
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.gethostbyname = _guarded_gethostbyname
    socket.gethostbyname_ex = _guarded_gethostbyname_ex


def uninstall_network_guard() -> None:
    """Put the original socket functions back."""
    socket.socket.connect = ORIGINALS.connect  # type: ignore[method-assign]
    socket.socket.connect_ex = ORIGINALS.connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = ORIGINALS.getaddrinfo
    socket.gethostbyname = ORIGINALS.gethostbyname
    socket.gethostbyname_ex = ORIGINALS.gethostbyname_ex


def database_url_is_local(url: str | None) -> bool:
    """Return True if url is unset, empty, or names a local database host.

    Local means the host is localhost, 127.0.0.1, or ::1 (IPv6 in brackets), or
    there is no host at all, which libpq treats as a local unix socket (for example
    postgresql:///filingcheck or postgresql://user@/db?host=/var/run/postgresql).
    Only the host is checked, never the username or password, so
    postgresql://localhost@db.example.com/x is refused. An unparseable URL is refused.
    """
    if url is None or url.strip() == "":
        return True
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return False
    return host is None or host == "" or host in DATABASE_URL_LOCAL_HOSTS


def _url_host(url: str | None) -> str | None:
    """Return the host part of url for error messages, or None if it cannot be parsed."""
    try:
        return urlsplit((url or "").strip()).hostname
    except ValueError:
        return None


def pytest_configure(config: pytest.Config) -> None:
    """Refuse a non-local DATABASE_URL, then install the guard before collection."""
    url = os.environ.get("DATABASE_URL")
    if not database_url_is_local(url):
        # Name only the host, so a password in the URL is never printed.
        raise pytest.UsageError(
            f"Refusing to run tests: DATABASE_URL points at non-local host {_url_host(url)!r}. "
            "Tests may only use a database on localhost, 127.0.0.1, ::1, or a local unix "
            "socket. Unset DATABASE_URL or point it at the local docker-compose Postgres."
        )
    install_network_guard()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the guard when the pytest session ends."""
    uninstall_network_guard()


@pytest.fixture
def network_guard() -> SimpleNamespace:
    """Give tests access to the guard's error class and the original functions."""
    return SimpleNamespace(
        error=NetworkBlockedError,
        originals=ORIGINALS,
        is_local_host=is_local_host,
        database_url_is_local=database_url_is_local,
    )
