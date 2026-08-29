# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_llamacpp_image_is_pinned_and_verifies_its_wheel() -> None:
    dockerfile = (
        Path(__file__).parents[3] / "docker" / "Dockerfile.llamacpp"
    ).read_text()

    assert "python:3.12-slim@sha256:" in dockerfile
    assert "latest" not in dockerfile
    wheel = "/tmp/lmcache-0.5.5.dev53-cp312-cp312-linux_x86_64.whl"
    assert f'echo "${{LMCACHE_WHEEL_SHA256}}  {wheel}"' in dockerfile
    assert f"COPY ${{LMCACHE_WHEEL}} {wheel}" in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert "--constraint /tmp/llamacpp-constraints.txt" in dockerfile
    assert (
        'ENTRYPOINT ["python", "-m", '
        '"lmcache.integration.llamacpp.checkpoint_daemon"]' in dockerfile
    )
