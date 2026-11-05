#!/usr/bin/env bash
# Build the service, prepare each startup mode and measure it.
# Usage: ./run.sh [modes...]   default: jvm cds aot aot-train
#   crac   needs a CRaC-enabled JDK (Linux), native needs GraalVM in JAVA_HOME.
#   RUNS=7 ./run.sh             number of measured runs per mode (default 5)
#   REQUIRE_ALL=1 ./run.sh ...  fail if a mode from expected.txt was not measured
set -euo pipefail
cd "$(dirname "$0")"
[ -n "${JAVA_HOME:-}" ] && export PATH="$JAVA_HOME/bin:$PATH"

MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(jvm cds aot aot-train)
RUNS="${RUNS:-5}"

needs_jar=0; needs_native=0
for m in "${MODES[@]}"; do
  if [ "$m" = native ]; then needs_native=1; else needs_jar=1; fi
done

if [ $needs_jar -eq 1 ]; then
  mvn -B -q -DskipTests package
  rm -rf build/app
  java -Djarmode=tools -jar target/app.jar extract --destination build/app
fi
if [ $needs_native -eq 1 ]; then
  mvn -B -q -Pnative -DskipTests native:compile
fi

for m in "${MODES[@]}"; do
  python3 bench.py "$m" --runs "$RUNS"
done

python3 report.py results ${REQUIRE_ALL:+--require-all}
