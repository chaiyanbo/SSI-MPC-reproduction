#!/usr/bin/env bash

SSI_MPC_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SSI_MPC_CATKIN_WS="/workspaces/ssi_mpc_ws"
ACADOS_PINNED_DIR="/workspaces/acados-v0.4.4"

source /opt/ros/noetic/setup.bash
source "${SSI_MPC_CATKIN_WS}/devel/setup.bash"
source "${SSI_MPC_CATKIN_WS}/.venv/bin/activate"

export ACADOS_SOURCE_DIR="${ACADOS_PINNED_DIR}"

case ":${LD_LIBRARY_PATH:-}:" in
  *":${ACADOS_PINNED_DIR}/lib:"*) ;;
  *) export LD_LIBRARY_PATH="${ACADOS_PINNED_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

case ":${PYTHONPATH:-}:" in
  *":${SSI_MPC_REPO_DIR}/ros_mpc:"*) ;;
  *) export PYTHONPATH="${SSI_MPC_REPO_DIR}/ros_mpc${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

unset SSI_MPC_REPO_DIR SSI_MPC_CATKIN_WS ACADOS_PINNED_DIR
