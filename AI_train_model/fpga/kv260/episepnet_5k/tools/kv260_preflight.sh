#!/usr/bin/env bash
# Capture reproducible, non-secret build-host or KV260-target facts.
set -u
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ai_train_root="$(cd "${script_dir}/../../../.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
host_short="$(hostname -s 2>/dev/null || hostname)"
output_dir="${1:-${ai_train_root}/fpga/kv260/episepnet_5k/runs/preflight_${host_short}_${timestamp}}"

mkdir -p "${output_dir}"

capture() {
    local name="$1"
    shift
    {
        printf '$ '
        printf ' %q' "$@"
        printf '\n'
        "$@"
        local status=$?
        printf '\nexit_status=%s\n' "${status}"
        return 0
    } > "${output_dir}/${name}.txt" 2>&1
}

capture_shell() {
    local name="$1"
    local command="$2"
    capture "${name}" bash -lc "${command}"
}

probe_tool() {
    local tool="$1"
    local version_args="$2"
    capture_shell "tool_${tool}" "if command -v ${tool} >/dev/null 2>&1; then command -v ${tool}; ${tool} ${version_args} || true; else echo MISSING; fi"
}

machine="$(uname -m)"
role="build_host"
if [[ "${machine}" == "aarch64" || "${machine}" == "arm64" ]]; then
    role="kv260_target_candidate"
fi

{
    printf 'role=%s\n' "${role}"
    printf 'hostname=%s\n' "${host_short}"
    printf 'machine=%s\n' "${machine}"
    printf 'timestamp_utc=%s\n' "${timestamp}"
    printf 'ai_train_root=%s\n' "${ai_train_root}"
} > "${output_dir}/summary.env"

capture system_uname uname -a
capture_shell system_os_release 'cat /etc/os-release 2>/dev/null || true'
capture system_lscpu lscpu
capture system_memory free -h
capture system_storage df -h
capture_shell system_network 'ip -br link 2>/dev/null || true'
capture_shell system_pci 'lspci -nn 2>/dev/null || true'
capture_shell system_accel_devices 'ls -l /dev/dri /dev/xclmgmt /dev/xocl 2>/dev/null || true'
capture_shell system_xilinx_env 'env | grep -E "^(PATH|XILINX|XRT|PLATFORM|VITIS|PETALINUX)=" || true'

probe_tool vitis_hls '-version'
probe_tool vivado '-version'
probe_tool v++ '--version'
probe_tool xsct '-version'
probe_tool xbutil '--version'
probe_tool xrt-smi '--version'
probe_tool xmutil '--help'

capture_shell xrt_xbutil_examine 'if command -v xbutil >/dev/null 2>&1; then xbutil examine; else echo MISSING; fi'
capture_shell xrt_xbutil_electrical 'if command -v xbutil >/dev/null 2>&1; then xbutil examine --report electrical; else echo MISSING; fi'
capture_shell xrt_smi_examine 'if command -v xrt-smi >/dev/null 2>&1; then xrt-smi examine; else echo MISSING; fi'
capture_shell kv260_xmutil_packages 'if command -v xmutil >/dev/null 2>&1; then xmutil getpkgs; else echo MISSING; fi'

package_dir="${ai_train_root}/fpga/reference_run_21_int16"
capture_shell package_ledger "if test -d '${package_dir}' && command -v sha256sum >/dev/null 2>&1; then sha256sum '${package_dir}/model_manifest.json' '${package_dir}/normalization.json' '${package_dir}'/tensors/* '${package_dir}'/test_vectors/*; else echo PACKAGE_OR_SHA256SUM_MISSING; fi"

printf 'KV260 preflight captured in %s\n' "${output_dir}"
