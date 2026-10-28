#!/usr/bin/env python3
"""Launch the two-node H100 recipes using their explicit runtime settings."""

if __package__:
    from .run_baseline import main
else:
    from run_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
