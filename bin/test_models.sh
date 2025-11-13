MODULE_HOME="$(dirname "$(dirname "$(readlink -f "$0")")")"

set -a && source "$MODULE_HOME/conf/agent.conf" && set +a

python "$MODULE_HOME/test/test_models.py"