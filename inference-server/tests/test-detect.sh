#!/usr/bin/env bash
# detect.sh 栈决策门禁测试（无需 GPU：用假 nvidia-smi 注入探测结果）
#
# 为什么必须有这个测试：20/30/40/50 系与老驱动组合无法在现役机器上覆盖，
# 判定错一次 = 用户在真机上装出跑不起来的栈（no kernel image / 驱动不足）。
#
# 负例自检：把 scripts/detect.sh 里 pre-Turing 分支删掉，case "GTX 1080" 必须失败。
#
# 用法：bash tests/test-detect.sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DETECT="$HERE/scripts/detect.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n    期望 stack=%s，实际 stack=%s\n' "$1" "$2" "$3"; }

# 假 nvidia-smi：FAKE_CSV_CC = 带 compute_cap 的输出；FAKE_CSV_NOCC = 老驱动字段缺失
mkdir -p "$TMP/bin"
cat > "$TMP/bin/nvidia-smi" <<'FAKE'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in
  *compute_cap*)
    [ -n "${FAKE_CSV_CC:-}" ] || { echo "Field \"compute_cap\" is not a valid field to query." >&2; exit 2; }
    printf '%s\n' "$FAKE_CSV_CC"; exit 0 ;;
esac; done
printf '%s\n' "${FAKE_CSV_NOCC:-}"
FAKE
chmod +x "$TMP/bin/nvidia-smi"

# run_case <描述> <期望栈> <CC(带字段)> <老格式>
#   自动把 driver 从 CSV 里取出后的期望最低驱动也校验
run_case() {
  local desc=$1 want=$2 csv_cc=$3 csv_nocc=${4:-}
  local out stack code
  out=$(PATH="$TMP/bin:$PATH" FAKE_CSV_CC="$csv_cc" FAKE_CSV_NOCC="$csv_nocc" \
        bash "$DETECT" 2>/dev/null); code=$?
  stack=$(printf '%s\n' "$out" | awk -F= '/^STACK=/{print $2}')
  if [ "$stack" = "$want" ]; then
    # 期望栈为 X 时退出码必须是 2
    if [ "$want" = "X" ] && [ "$code" -ne 2 ]; then
      bad "$desc（X 档退出码应为 2，实际 $code）" "$want" "$stack"; return
    fi
    [ "$want" != "X" ] && [ "$code" -ne 0 ] && { bad "$desc（可用栈退出码应为 0，实际 $code）" "$want" "$stack"; return; }
    ok "$desc → $want"
  else
    bad "$desc" "$want" "${stack:-<空>}"
  fi
}

echo "== 10 系 / Pascal =="
run_case "GTX 1080 sm_61 驱动582"      P      "NVIDIA GeForce GTX 1080, 6.1, 8192, 582.28"
run_case "GTX 1080 sm_61 驱动470"      P-lite "NVIDIA GeForce GTX 1080, 6.1, 8192, 470.82"
run_case "GTX 1080 sm_61 驱动440"      X      "NVIDIA GeForce GTX 1080, 6.1, 8192, 440.31"
run_case "Tesla P40 sm_61 驱动582"     P      "Tesla P40, 6.1, 22919, 582.28"

echo "== Volta（CUDA 13 已移除，必须旧栈）=="
run_case "V100 sm_70 驱动582（不算 modern）" P  "Tesla V100-SXM2-16GB, 7.0, 16160, 582.28"

echo "== 20/30/40/50 系 =="
run_case "RTX 2080Ti sm_75 驱动582"    M      "NVIDIA GeForce RTX 2080 Ti, 7.5, 11264, 582.28"
run_case "GTX 1660 sm_75 驱动550（降级）" P    "NVIDIA GeForce GTX 1660, 7.5, 6144, 550.54"
run_case "RTX 3090 sm_86 驱动582"      M      "NVIDIA GeForce RTX 3090, 8.6, 24576, 582.28"
run_case "RTX 3090 sm_86 驱动570（降级）" P    "NVIDIA GeForce RTX 3090, 8.6, 24576, 570.26"
run_case "A100 sm_80 驱动582"          M      "NVIDIA A100-SXM4-40GB, 8.0, 40960, 582.28"
run_case "RTX 4090 sm_89 驱动582"      M      "NVIDIA GeForce RTX 4090, 8.9, 24564, 582.28"
run_case "RTX 5090 sm_120 驱动582"     M      "NVIDIA GeForce RTX 5090, 12.0, 32760, 582.28"
run_case "RTX 5090 sm_120 驱动550（无解）" X   "NVIDIA GeForce RTX 5090, 12.0, 32760, 550.54"

echo "== 不支持的老架构 =="
run_case "GTX 970 Maxwell sm_52"       X      "NVIDIA GeForce GTX 970, 5.2, 4096, 582.28"

echo "== compute_cap 字段缺失（老驱动）时按型号推断 =="
run_case "老驱动无 CC 字段：RTX 3060"   M      ""  "NVIDIA GeForce RTX 3060, 12288, 582.28"
run_case "老驱动无 CC 字段：RTX 4060"   M      ""  "NVIDIA GeForce RTX 4060, 8188, 582.28"
run_case "老驱动无 CC 字段：GTX 1080 驱动470" P-lite ""  "NVIDIA GeForce GTX 1080, 8192, 470.82"
run_case "老驱动无 CC 字段：RTX 5070"   M      ""  "NVIDIA GeForce RTX 5070, 12227, 582.28"
run_case "老驱动无 CC 字段：RTX 2060 驱动470" P-lite ""  "NVIDIA GeForce RTX 2060, 6144, 470.82"

echo "== 多卡：按算力最低的定栈 =="
MULTI="NVIDIA GeForce RTX 4090, 8.9, 24564, 582.28
NVIDIA GeForce GTX 1080, 6.1, 8192, 582.28"
out=$(PATH="$TMP/bin:$PATH" FAKE_CSV_CC="$MULTI" bash "$DETECT" 2>/dev/null)
stack=$(printf '%s\n' "$out" | awk -F= '/^STACK=/{print $2}')
count=$(printf '%s\n' "$out" | awk -F= '/^GPU_COUNT=/{print $2}')
[ "$stack" = "P" ] && [ "$count" = "2" ] && ok "4090+1080 混插 → P（共 2 张）" || bad "4090+1080 混插" "P/2" "$stack/$count"

echo
printf '== 结果：%d 通过，%d 失败 ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
