# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOWS = (
    ROOT / ".github/workflows/unit-tests.yml",
    ROOT / ".github/workflows/full-test-suite.yml",
)
OSWORLD_AGENT_REQUIREMENTS = ROOT / "responses_api_agents/osworld_agent/requirements.txt"
OSWORLD_RESOURCES_PROJECT = ROOT / "resources_servers/osworld/pyproject.toml"


def _uv_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["uv"]


def test_uv_version_supports_scoped_dependency_exclusions() -> None:
    required_version = SpecifierSet(_uv_config()["required-version"])

    assert Version("0.11.24") not in required_version
    assert Version("0.11.25") in required_version


def test_uv_dependency_exclusions_are_scoped() -> None:
    exclusions = _uv_config()["exclude-dependencies"]

    assert exclusions
    assert all(isinstance(exclusion, dict) for exclusion in exclusions), (
        "Global string exclusions silently remove direct server requirements; "
        "scope every exclusion to the package that declares the unwanted dependency edge."
    )


def test_ci_uv_versions_satisfy_project_minimum() -> None:
    required_version = SpecifierSet(_uv_config()["required-version"])

    for workflow in CI_WORKFLOWS:
        match = re.search(r"astral\.sh/uv/([^/]+)/install\.sh", workflow.read_text())
        assert match, f"Missing pinned uv installer in {workflow}"
        assert Version(match.group(1)) in required_version


def test_mlflow_keeps_shared_cryptography_edge() -> None:
    exclusions = _uv_config()["exclude-dependencies"]
    mlflow = next(exclusion for exclusion in exclusions if exclusion["package"]["name"] == "mlflow")

    assert "cryptography" not in mlflow["dependencies"]


def test_osworld_runtime_consumers_share_one_pinned_revision() -> None:
    revision_pattern = re.compile(r"OSWorld/archive/([0-9a-f]{40})\.tar\.gz")
    agent_match = revision_pattern.search(OSWORLD_AGENT_REQUIREMENTS.read_text(encoding="utf-8"))
    resources_match = revision_pattern.search(OSWORLD_RESOURCES_PROJECT.read_text(encoding="utf-8"))

    assert agent_match, "The OSWorld agent must pin an immutable OSWorld archive revision"
    assert resources_match, "The OSWorld Resources Server must pin an immutable OSWorld archive revision"
    assert agent_match.group(1) == resources_match.group(1)
