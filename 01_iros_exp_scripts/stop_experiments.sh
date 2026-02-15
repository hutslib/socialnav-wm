#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALCON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FORCE=0
ASSUME_YES=0
FORCE_IF_NEEDED=1

usage() {
    echo "Stop Falcon training processes started by experiment scripts."
    echo ""
    echo "Usage:"
    echo "  ./stop_experiments.sh             # graceful stop (SIGTERM)"
    echo "  ./stop_experiments.sh --force     # force stop (SIGKILL)"
    echo "  ./stop_experiments.sh -y          # no confirmation"
    echo ""
    echo "Options:"
    echo "  -f, --force    send SIGKILL instead of SIGTERM"
    echo "  -y, --yes      skip confirmation prompt"
    echo "  --no-auto-force disable auto SIGKILL for remaining processes"
    echo "  -h, --help     show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force)
            FORCE=1
            shift
            ;;
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        --no-auto-force)
            FORCE_IF_NEEDED=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

PATTERNS=(
    "single_node_falcon.sh"
    "habitat-baselines/habitat_baselines/run.py"
    "torch.distributed.launch"
)

print_matches() {
    local found=1
    local printed_header=0
    local entries=()
    local pattern
    local line

    for pattern in "${PATTERNS[@]}"; do
        while IFS= read -r line; do
            [[ -n "$line" ]] && entries+=("$line")
        done < <(pgrep -af "$pattern" || true)
    done

    if [[ ${#entries[@]} -gt 0 ]]; then
        if [[ $printed_header -eq 0 ]]; then
            echo "Matched processes:"
            printed_header=1
        fi
        printf "%s\n" "${entries[@]}" | sort -u
        found=0
    fi
    return $found
}

collect_pids() {
    local pids=()
    local pattern
    for pattern in "${PATTERNS[@]}"; do
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && pids+=("$pid")
        done < <(pgrep -f "$pattern" || true)
    done

    if [[ ${#pids[@]} -eq 0 ]]; then
        return 1
    fi

    printf "%s\n" "${pids[@]}" | sort -u
    return 0
}

echo "Falcon root: $FALCON_ROOT"
echo ""

if ! print_matches; then
    echo "No matching training processes found."
    exit 0
fi

mapfile -t PIDS < <(collect_pids || true)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "No matching training processes found."
    exit 0
fi

echo ""
echo "Found ${#PIDS[@]} process(es): ${PIDS[*]}"

if [[ $ASSUME_YES -ne 1 ]]; then
    read -r -p "Proceed to stop them? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

if [[ $FORCE -eq 1 ]]; then
    echo "Sending SIGKILL..."
    kill -9 "${PIDS[@]}" || true
else
    echo "Sending SIGTERM..."
    kill -TERM "${PIDS[@]}" || true
    echo "Waiting 5 seconds..."
    sleep 5
fi

echo ""
echo "Remaining matched processes:"
if print_matches; then
    if [[ $FORCE -eq 0 && $FORCE_IF_NEEDED -eq 1 ]]; then
        mapfile -t REMAINING_PIDS < <(collect_pids || true)
        if [[ ${#REMAINING_PIDS[@]} -gt 0 ]]; then
            echo ""
            echo "Auto force-stopping remaining process(es): ${REMAINING_PIDS[*]}"
            kill -9 "${REMAINING_PIDS[@]}" || true
            sleep 1
            echo ""
            echo "Remaining matched processes after auto-force:"
            if print_matches; then
                echo "Some processes are still alive after auto-force."
                exit 1
            fi
            echo "All matched processes stopped."
            exit 0
        fi
    fi
    echo ""
    echo "Some processes are still alive."
    echo "Run again with --force if needed."
    exit 1
fi

echo "All matched processes stopped."
