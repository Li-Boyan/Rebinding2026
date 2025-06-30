"""
Usage:
    core_llps.py <config_dir> [--load-curr-state]

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from rebinding.micropk import run_micropk


def main():
    args = docopt(__doc__)
    run_micropk(args["<config_dir>"], args["--load-curr-state"])


if __name__ == "__main__":
    main()
