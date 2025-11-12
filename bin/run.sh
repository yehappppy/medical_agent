MODULE_HOME="$(dirname "$(dirname "$(readlink -f "$0")")")"

set -a && source $MODULE_HOME/conf/agent.conf && set +a

python -m agent.test