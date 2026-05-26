"""Command-line entry points for ``android_packer``.

每个子模块提供一个 ``main(argv: Optional[list[str]] = None) -> int`` 入口，
既可以通过 ``project.scripts`` 暴露为 ``android-packer-*`` 命令，也可以被
``scripts/`` 下的 thin wrapper 直接调用。
"""

from __future__ import annotations

__all__: list[str] = []
