#!/bin/bash
# Downloads the 23 NYC cellular traces used by exp/ton_3dgs.toml and
# exp/ton_gcc_convergence.toml into traces/nyc-<id>-<place>.trace.
#
# They are the mahimahi traces published with Orca (Abbasloo et al.,
# SIGCOMM 2020) in https://github.com/Soheil-ab/Cellular-Traces-NYC, file
# trace-<id>-<place>, saved under another name and not changed. Twelve of
# them are already in the repository; the script checks those and fetches
# the others. Every file must match the SHA-256 below, which is that of the
# files the paper's runs used.
#
#   bash traces/download_nyc.sh
set -euo pipefail
cd "$(dirname "$0")"
URL=https://raw.githubusercontent.com/Soheil-ab/Cellular-Traces-NYC/master

SUMS=$(cat <<'SHA'
07e5e74d1b480ec193d981b278f3e13751ceaea5d84f858ace1b19d6a4e13108  nyc-2767958-taxi1.trace
7d6ade5b4f9433e20ce675b57f342d911ac987e6faea2471959434ac46e1632e  nyc-2768760-taxi3.trace
dc66eddd1e570ae52e8795d7f8f0367ebf29cf64e6504d0442facca8f05cc138  nyc-2925703-home1.trace
be5064bb7496d2159d8bb1f7862063cf655e6eb473047b4f62f0891cc5b3738c  nyc-3093662-home3.trace
7252474afe1b0551d0a36f6772d4eeaa183801b8f8eeaae02ac41153a3445d67  nyc-3109898-bus.trace
37cd1d5f768d4ab80ced579498f292d82a898457896be7dd9a533d9cc332e0af  nyc-3114405-bus.trace
604ebe5756ed24e5f8bbaed55979c7731a926c76f2821e7b6a4ebc9647e091b6  nyc-3189663-timessquare.trace
91a22130d24b530f4a10c9491009e6e4ed2befbfe99f0070696ae2c1afc5f19e  nyc-3199718-timessquare.trace
2a903bf70999c70f7416a442f6eb03e0c9463aca62df0adb47b27b78c78de76e  nyc-3201037-timessquare.trace
9635f7806e64a00788a10c4c5738dcbf313f6f674ae6c86b70ec182c5fa3cc4a  nyc-3201711-timessquare.trace
b966e3c5e5507365e5ea32d408b51e49627768c246f12285c7052d1f737d17ee  nyc-3202253-timessquare.trace
690ba0e7522a1af8312e81b081656032acf61e50427c741e82b99ce56c8a17a3  nyc-3203600-timessquare.trace
d849220ff6bca82958e4b6907919e619535ecf99cb607f5d7f21ed3fe127b68d  nyc-3204803-timessquare.trace
9ad394714004cfe561109479b97a72704adca834f0ec7b8431b02ec096f0907d  nyc-3205967-timessquare.trace
32154b8cf18f250469dcb66358476251d78577c3eee8e8c1cdb8203ee3c988b1  nyc-3207521-timessquare.trace
48d7cc574142adff84b83fc6037c9c77d4fcfe5d746a05157a0525cbe7b63288  nyc-3208852-timessquare.trace
574d5c44bcb43a3d9a861e1f959b64da8a071cc69ab89458ce20678697fd04a7  nyc-3219061-home.trace
2a90dcba1547afbebc9643b332b70f43b1eef224815534a503351d4e7ad8b333  nyc-3453943-timessquare.trace
76e70ed7bea3173b141a4a83b4eb59bd6688e13560ada6fb988dd0b986fd11b2  nyc-3455408-timessquare.trace
953990b14d4129fcde97d2d09e51f7562d2e37afa40bab56056e2f2810f5e3de  nyc-3457194-timessquare.trace
9b7341ea305822c388563f94f8ecbde3a1c4d667f1bfeae4be144e21a547504f  nyc-3458374-timessquare.trace
65b55e2dbfbe8b9b9bc5e3d40026cd43790e7283ace4744a7622df76467acfbd  nyc-3552192-bus.trace
2d2d5d2a4002a3bbc7f9bf94a66379175b8bfb6ca9950d5e801e13bafa4868fb  nyc-3555076-bus.trace
SHA
)

while read -r sum name; do
    if [[ ! -f $name ]]; then
        raw=trace-${name#nyc-}
        raw=${raw%.trace}
        echo "downloading $name"
        curl -fsSL --retry 3 -o "$name.tmp" "$URL/$raw"
        mv "$name.tmp" "$name"
    fi
done <<<"$SUMS"
sha256sum -c --quiet <<<"$SUMS"
echo "all $(wc -l <<<"$SUMS") NYC traces present and verified"
