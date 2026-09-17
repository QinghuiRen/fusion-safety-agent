"""支持通过 ``python -m fusion_tracking_agent`` 调用产品命令行。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
