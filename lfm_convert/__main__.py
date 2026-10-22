from .cli import main
import sys

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("lfm: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"lfm: {error}", file=sys.stderr)
        raise SystemExit(2)
