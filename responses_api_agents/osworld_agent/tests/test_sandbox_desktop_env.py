from __future__ import annotations

from typing import Any

from responses_api_agents.osworld_agent import sandbox_desktop_env


def _capture_desktop_env_kwargs(monkeypatch: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_init(_self: Any, *_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sandbox_desktop_env.DesktopEnv, "__init__", fake_init)
    return captured


def test_opensandbox_pool_skips_local_docker_vm_resolution(monkeypatch: Any) -> None:
    captured = _capture_desktop_env_kwargs(monkeypatch)

    sandbox_desktop_env.SandboxDesktopEnv(
        sandbox_provider={"opensandbox": {"connection": {}}},
        sandbox_spec={
            "provider_options": {"extensions": {"poolRef": "osworld-kvm"}},
        },
        sandbox_require_kvm=False,
    )

    assert captured["path_to_vm"] == "opensandbox-pool-managed"


def test_explicit_vm_path_is_preserved_for_opensandbox(monkeypatch: Any) -> None:
    captured = _capture_desktop_env_kwargs(monkeypatch)

    sandbox_desktop_env.SandboxDesktopEnv(
        sandbox_provider={"opensandbox": {"connection": {}}},
        sandbox_spec={
            "provider_options": {"extensions": {"poolRef": "osworld-kvm"}},
        },
        sandbox_require_kvm=False,
        path_to_vm="caller-supplied-path",
    )

    assert captured["path_to_vm"] == "caller-supplied-path"


def test_docker_sandbox_does_not_receive_pool_sentinel(monkeypatch: Any) -> None:
    captured = _capture_desktop_env_kwargs(monkeypatch)

    sandbox_desktop_env.SandboxDesktopEnv(
        sandbox_provider={"docker": {}},
        sandbox_spec={"image": "docker://osworld@sha256:fixed"},
    )

    assert "path_to_vm" not in captured
