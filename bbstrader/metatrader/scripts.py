import argparse
import multiprocessing
import sys

from bbstrader.apps._copier import main as RunCopyApp
from bbstrader.metatrader.copier import RunCopier, config_copier, copier_worker_process


def copier_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="CLI",
        choices=("CLI", "GUI"),
        help="Run the copier in the terminal or using the GUI",
    )
    parser.add_argument(
        "-s", "--source", type=str, nargs="?", default=None, help="Source section name"
    )
    parser.add_argument(
        "-I", "--id", type=int, default=0, help="Source Account unique ID"
    )
    parser.add_argument(
        "-U",
        "--unique",
        action="store_true",
        help="Specify if the source account is only master",
    )
    parser.add_argument(
        "-d",
        "--destinations",
        type=str,
        nargs="*",
        default=None,
        help="Destination section names",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.1, help="Update interval in seconds"
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        default=None,
        type=str,
        help="Config file name or path",
    )
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        nargs="?",
        default=None,
        help="Start time in HH:MM format",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        nargs="?",
        default=None,
        help="End time in HH:MM format",
    )
    parser.add_argument(
        "-M",
        "--multiprocess",
        action="store_true",
        help="Run each destination account in a separate process.",
    )
    return parser


def copy_trades(unknown):
    HELP_MSG = """
    Usage:
        python -m bbstrader --run copier [options]

    Options:
        -m, --mode: CLI for terminal app and GUI for Desktop app
        -s, --source: Source Account section name
        -I, --id: Source Account unique ID
        -U, --unique: Specify if the source account is only master 
        -d, --destinations: Destination Account section names (multiple allowed)
        -i, --interval: Update interval in seconds
        -M, --multiprocess: When set to True, each destination account runs in a separate process.
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

    if copy_args.mode == "GUI":
        RunCopyApp()

    elif copy_args.mode == "CLI":
        source, destinations = config_copier(
            source_section=copy_args.source,
            dest_sections=copy_args.destinations,
            inifile=copy_args.config,
        )
        source["id"] = copy_args.id
        source["unique"] = copy_args.unique
        if copy_args.multiprocess:
            copier_processes = []
            for dest_config in destinations:
                process = multiprocessing.Process(
                    target=copier_worker_process,
                    args=(
                        source,
                        dest_config,
                        copy_args.interval,
                        copy_args.start,
                        copy_args.end,
                    ),
                )
                process.start()
                copier_processes.append(process)
            for process in copier_processes:
                process.join()
        else:
            RunCopier(
                source,
                destinations,
                copy_args.interval,
                copy_args.start,
                copy_args.end,
            )
