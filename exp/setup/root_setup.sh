#!/usr/bin/env bash
# Steps of the experiment setup that need root. Run once, and again after a
# reboot (the CPU governor does not persist):
#
#   sudo bash exp/setup/root_setup.sh /path/to/rattan
#
# 1. installs the given rattan binary as /usr/local/bin/rattan and grants it
#    the capabilities `rattan link` needs (cp/install drop file capabilities,
#    so setcap has to run after every replacement);
# 2. loads the veth module;
# 3. sets every CPU to the performance governor, so that render and encode
#    times do not depend on frequency scaling;
# 4. if ufw is on, lets the rattan namespace reach the host. In compatible
#    mode the client side gets 10.1.x.x and the host end 10.2.x.x; ufw drops
#    their traffic otherwise (TIGAS needs UDP 4433, check_env.sh TCP 5298-5299).
#    The rule only matches these veth addresses, nothing from outside.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo" >&2
    exit 1
fi
SRC=${1:?usage: sudo bash root_setup.sh /path/to/rattan}
DST=/usr/local/bin/rattan
CAPS='cap_dac_override,cap_dac_read_search,cap_sys_ptrace,cap_net_admin,cap_sys_admin,cap_net_raw+ep'

echo "== rattan"
if [[ -e $DST ]] && ! cmp -s "$SRC" "$DST"; then
    bak=$DST.bak.$(date +%Y%m%d%H%M%S)
    cp -p "$DST" "$bak"
    echo "old binary kept as $bak"
fi
install -m 0755 "$SRC" "$DST"
setcap "$CAPS" "$DST"
getcap "$DST"
"$DST" --version | head -3

echo "== veth"
modprobe veth
lsmod | grep -w veth

echo "== CPU governor"
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done
sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c

echo "== ufw"
if command -v ufw >/dev/null && ufw status | grep -q 'Status: active'; then
    ufw allow from 10.1.0.0/16 to 10.2.0.0/16 comment 'rattan link'
    ufw status | grep -E '10\.1\.0\.0/16'
else
    echo "ufw not active, nothing to do"
fi

echo "done"
