# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from resources_servers.browsergym_web import evaluator_compat


def test_webarena_model_override_uses_its_provider_and_preserves_arguments(monkeypatch):
    calls = []
    imports = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "same"

    provider = SimpleNamespace(generate_from_openai_chat_completion=original)
    helpers = SimpleNamespace(generate_from_openai_chat_completion=original)

    def fake_import(name):
        imports.append(name)
        if name == "webarena.llms.providers.openai_utils":
            return provider
        if name == "webarena.evaluation_harness.helper_functions":
            return helpers
        raise AssertionError(name)

    monkeypatch.setattr(evaluator_compat.importlib, "import_module", fake_import)
    evaluator_compat.configure_webarena_evaluator_model("local-judge")

    assert helpers.generate_from_openai_chat_completion([], "gpt-4-1106-preview", 0, 768, 1.0, 0) == "same"
    assert imports == [
        "webarena.llms.providers.openai_utils",
        "webarena.evaluation_harness.helper_functions",
    ]
    assert calls == [(([], "local-judge", 0, 768, 1.0, 0), {})]


def test_visualwebarena_model_override_preserves_prompt_and_generation_options(monkeypatch):
    calls = []
    imports = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "correct"

    provider = SimpleNamespace(generate_from_openai_chat_completion=original)
    helpers = SimpleNamespace(generate_from_openai_chat_completion=original)

    def fake_import(name):
        imports.append(name)
        if name == "visualwebarena.llms.providers.openai_utils":
            return provider
        if name == "visualwebarena.evaluation_harness.helper_functions":
            return helpers
        raise AssertionError(name)

    monkeypatch.setattr(evaluator_compat.importlib, "import_module", fake_import)
    evaluator_compat.configure_visualwebarena_evaluator_model("local-visual-judge")

    answer = helpers.generate_from_openai_chat_completion(
        messages=[{"role": "user", "content": "grade this"}],
        model="gpt-4-1106-preview",
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    )

    assert answer == "correct"
    assert imports == [
        "visualwebarena.llms.providers.openai_utils",
        "visualwebarena.evaluation_harness.helper_functions",
    ]
    assert calls[0][1]["model"] == "local-visual-judge"


def test_evaluator_environment_supports_openai_v0_and_v1(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    evaluator_compat.configure_evaluator_environment(
        api_key="test-only",  # pragma: allowlist secret
        base_url="http://judge.test/v1",
    )

    assert evaluator_compat.os.environ["OPENAI_API_KEY"] == "test-only"  # pragma: allowlist secret
    assert evaluator_compat.os.environ["OPENAI_BASE_URL"] == "http://judge.test/v1"
    assert evaluator_compat.os.environ["OPENAI_API_BASE"] == "http://judge.test/v1"


def test_rule_only_import_environment_restores_process_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")  # pragma: allowlist secret
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://existing.test/v1")

    with evaluator_compat.rule_only_evaluator_import_environment(
        base_url="http://temporary.test/v1",
    ):
        assert (
            evaluator_compat.os.environ["OPENAI_API_KEY"]
            == "unused-for-rule-only-evaluator"  # pragma: allowlist secret
        )
        assert evaluator_compat.os.environ["OPENAI_BASE_URL"] == "http://temporary.test/v1"
        assert evaluator_compat.os.environ["OPENAI_API_BASE"] == "http://temporary.test/v1"

    assert evaluator_compat.os.environ["OPENAI_API_KEY"] == "existing-key"  # pragma: allowlist secret
    assert "OPENAI_BASE_URL" not in evaluator_compat.os.environ
    assert evaluator_compat.os.environ["OPENAI_API_BASE"] == "http://existing.test/v1"
