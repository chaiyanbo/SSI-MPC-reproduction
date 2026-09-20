FROM osrf/ros:noetic-desktop-full-focal
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl git git-lfs ninja-build openssh-server pkg-config python3-catkin-tools python3-dev python3-pip python3-rosdep python3-setuptools python3-venv python3-vcstool sudo wget && apt-get clean
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
WORKDIR /workspaces
