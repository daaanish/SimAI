#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")
ROOT_DIR=$(realpath "${SCRIPT_DIR:?}"/..)
NS3_DIR="${ROOT_DIR:?}"/ns-3-alibabacloud
SIMAI_DIR="${ROOT_DIR:?}"/astra-sim-alibabacloud
SOURCE_NS3_BIN_DIR="${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/simulation/build/scratch/ns3.36.1-AstraSimNetwork-debug
SOURCE_ANA_BIN_DIR="${SIMAI_DIR:?}"/build/simai_analytical/build/simai_analytical/SimAI_analytical
SOURCE_PHY_BIN_DIR="${SIMAI_DIR:?}"/build/simai_phy/build/simai_phynet/SimAI_phynet

TARGET_BIN_DIR="${SCRIPT_DIR:?}"/../bin
SIMAI_NS3_DIR="${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/simulation

function normalize_ns3_suite {
    local suite="$1"
    case "$suite" in
    "point-to-point")
        printf -- "devices-point-to-point\n";;
    *)
        printf -- "%s\n" "$suite";;
    esac
}

function compile {
    local option="$1"
    local extra_arg="$2"
    case "$option" in
    "ns3")
        if [ -n "$extra_arg" ]; then
            printf -- "ns3 compile mode does not accept a test suite.\n" >&2
            printf -- "Use ./scripts/build.sh -t ns3 %s instead.\n" "$extra_arg" >&2
            return 1
        fi
        mkdir -p "${TARGET_BIN_DIR:?}"
        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_simulator" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_simulator
        fi
        mkdir -p "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
        cp -r "${NS3_DIR:?}"/* "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3
        ./build.sh -c ns3    
        ln -s "${SOURCE_NS3_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_simulator;;
    "phy")
        mkdir -p "${TARGET_BIN_DIR:?}"
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy
        ./build.sh -c phy 
        ln -s "${SOURCE_PHY_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_phynet;;
    "analytical")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical
        ./build.sh -c analytical 
        ln -s "${SOURCE_ANA_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_analytical;;
    esac
}

function run_tests {
    local option="$1"
    local suite="$2"
    case "$option" in
    "ns3")
        if [ -z "$suite" ]; then
            printf -- "ns3 test mode requires a suite name.\n" >&2
            printf -- "Example: ./scripts/build.sh -t ns3 devices-point-to-point\n" >&2
            return 1
        fi
        if [ ! -d "${SIMAI_NS3_DIR:?}" ]; then
            printf -- "ns-3 simulation directory not found: %s\n" "${SIMAI_NS3_DIR:?}" >&2
            return 1
        fi
        suite=$(normalize_ns3_suite "$suite")
        cd "${SIMAI_NS3_DIR:?}" || return 1
        ./ns3 configure --enable-tests || return 1
        ./ns3 build || return 1
        python3 test.py --suite="$suite";;
    *)
        printf -- "test mode supported ns3 only (example:./build.sh -t ns3 devices-point-to-point)\n" >&2
        return 1;;
    esac
}

function cleanup_build {
    local option="$1"
    case "$option" in
    "ns3")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_simulator" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_simulator
        fi
        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3;;
    "phy")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy;;
    "analytical")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical;;
    esac
}

# Main Script
case "$1" in
-l|--clean)
    cleanup_build "$2";;
-c|--compile)
    compile "$2" "$3";;
-t|--test)
    run_tests "$2" "$3";;
-h|--help|*)
    printf -- "help message\n"
    printf -- "-c|--compile mode supported ns3/phy/analytical  (example:./build.sh -c ns3)\n"
    printf -- "-t|--test mode supported ns3  (example:./build.sh -t ns3 devices-point-to-point)\n"
    printf -- "-l|--clean  (example:./build.sh -l ns3)\n"
    printf -- "-lr|--clean-result mode  (example:./build.sh -lr ns3)\n"
esac
