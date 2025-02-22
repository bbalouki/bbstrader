import argparse
import sys

from bbstrader.metatrader.copier import RunCopier, config_copier


def copier_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-S", "--source", type=str, nargs="?", default=None, help="Source section name"
    )
    parser.add_argument(
        "-D",
        "--destinations",
        type=str,
        nargs="*",
        default=None,
        help="Destination section names",
    )
    parser.add_argument(
        "-I", "--interval", type=float, default=0.1, help="Update interval in seconds"
    )
    parser.add_argument(
        "-C",
        "--config",
        nargs="?",
        default=None,
        type=str,
        help="Config file name or path",
    )
    parser.add_argument(
        "-T",
        "--start",
        type=str,
        nargs="?",
        default=None,
        help="Start time in HH:MM format",
    )
    parser.add_argument(
        "-E",
        "--end",
        type=str,
        nargs="?",
        default=None,
        help="End time in HH:MM format",
    )
    return parser


def copy_trades(unknown):
    HELP_MSG = """
    Usage:
        python -m bbstrader --run copier [options]

    Options:
        -s, --source: Source Account section name
        -d, --destinations: Destination Account section names (multiple allowed)
        -i, --interval: Update interval in seconds
        -c, --config: .ini file or path (default: ~/.bbstrader/copier/copier.ini)
        -t, --start: Start time in HH:MM format
        -e, --end: End time in HH:MM format
    """
    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    copy_parser = argparse.ArgumentParser("Trades Copier", add_help=False)
    copy_parser = copier_args(copy_parser)
    copy_args = copy_parser.parse_args(unknown)

    source, destinations = config_copier(
        source_section=copy_args.source,
        dest_sections=copy_args.destinations,
        inifile=copy_args.config,
    )
    RunCopier(
        source,
        destinations,
        copy_args.interval,
        copy_args.start,
        copy_args.end,
    )
