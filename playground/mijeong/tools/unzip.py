import sys
import zipfile


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "-Z1":
        with zipfile.ZipFile(args[1]) as zf:
            sys.stdout.write("\n".join(zf.namelist()))
            sys.stdout.write("\n")
        return
    if len(args) == 3 and args[0] == "-p":
        with zipfile.ZipFile(args[1]) as zf:
            sys.stdout.buffer.write(zf.read(args[2]))
        return
    raise SystemExit(f"Unsupported unzip shim args: {args!r}")


if __name__ == "__main__":
    main()
