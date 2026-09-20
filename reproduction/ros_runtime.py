#!/usr/bin/env python3
"""ROS/Gazebo process and flight-safety boundary for one benchmark run."""

import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass

import rospy
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty


@dataclass(frozen=True)
class FlightOutcome:
    status: str
    completed: bool
    paused: bool
    final_lock_confirmed: bool
    error: str = ""


class ManagedProcess:
    """Own one subprocess, its process group, and its log file."""

    def __init__(self, name, command, log_path):
        self.name = name
        self.command = command
        self.log_path = log_path
        self.log_file = log_path.open("w", encoding="utf-8", buffering=1)
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self.log_file.close()
            raise
        self.process_group = self.process.pid

    @property
    def returncode(self):
        return self.process.poll()

    def _signal_group(self, requested_signal):
        try:
            os.killpg(self.process_group, requested_signal)
        except ProcessLookupError:
            return False
        return True

    def _group_exists(self):
        try:
            os.killpg(self.process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_for_group(self, timeout):
        deadline = time.monotonic() + timeout
        while self._group_exists() and time.monotonic() < deadline:
            self.process.poll()
            time.sleep(0.1)
        self.process.poll()
        return not self._group_exists()

    def stop(self):
        print(f"Stopping {self.name}", flush=True)
        try:
            for requested_signal, timeout in (
                (signal.SIGINT, 10.0),
                (signal.SIGTERM, 5.0),
                (signal.SIGKILL, 2.0),
            ):
                if not self._group_exists():
                    break
                self._signal_group(requested_signal)
                if self._wait_for_group(timeout):
                    break
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            self.log_file.close()


class FlightRunner:
    """Run one trajectory with explicit arming, monitoring, and cleanup."""

    def __init__(
        self,
        controller_command,
        reference_command,
        controller_log,
        reference_log,
        radius,
        altitude,
        reference_speed,
        flight_timeout,
    ):
        self.controller_command = controller_command
        self.reference_command = reference_command
        self.controller_log = controller_log
        self.reference_log = reference_log
        self.flight_timeout = flight_timeout

        self.horizontal_limit = max(2.5, 2.0 * radius + 0.5)
        self.altitude_limit = max(2.0, altitude + 1.0)
        self.speed_limit = max(4.0, 2.0 * reference_speed + 2.0)
        self.airborne_threshold = max(0.18, min(0.5, 0.5 * altitude))

        self.odometry = None
        self.odometry_wall = 0.0
        self.motor_speeds = None
        self.motor_wall = 0.0
        self.busy = False
        self.busy_received = False
        self.accept_window = False
        self.accepted = False
        self.paused = False

        self.controller = None
        self.reference = None

        rospy.init_node(
            "ssi_mpc_reproduction_benchmark",
            disable_signals=True,
        )
        self.arm_publisher = rospy.Publisher(
            "/hummingbird/bridge/arm",
            Bool,
            queue_size=1,
            latch=True,
        )
        self.subscribers = (
            rospy.Subscriber(
                "/hummingbird/ground_truth/odometry",
                Odometry,
                self._odometry_callback,
                queue_size=1,
            ),
            rospy.Subscriber("/busy", Bool, self._busy_callback, queue_size=1),
            rospy.Subscriber(
                "/hummingbird/command/motor_speed",
                Actuators,
                self._motor_callback,
                queue_size=1,
            ),
        )

    def _odometry_callback(self, message):
        self.odometry = message
        self.odometry_wall = time.monotonic()

    def _motor_callback(self, message):
        self.motor_speeds = tuple(message.angular_velocities)
        self.motor_wall = time.monotonic()

    def _busy_callback(self, message):
        self.busy = bool(message.data)
        self.busy_received = True
        if self.accept_window and self.busy:
            self.accepted = True

    def _publish_arm(self, value):
        message = Bool(data=value)
        for _ in range(10):
            self.arm_publisher.publish(message)
            time.sleep(0.05)

    def _motors_are_zero(self):
        if self.motor_speeds is None:
            return False
        if time.monotonic() - self.motor_wall > 2.0:
            return False
        return all(abs(value) <= 1e-6 for value in self.motor_speeds)

    def lock_and_confirm(self, timeout=3.0):
        try:
            self._publish_arm(False)
        except Exception as exception:
            print(f"INTERFACE LOCK PUBLISH FAILED: {exception}", flush=True)
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._motors_are_zero():
                print("INTERFACE LOCKED - MOTORS ZERO", flush=True)
                return True
            time.sleep(0.1)
        print("INTERFACE LOCK FAILED OR MOTOR STATE UNAVAILABLE", flush=True)
        return False

    def arm(self):
        self._publish_arm(True)
        print("INTERFACE ARMED", flush=True)

    def pause_gazebo(self):
        if self.paused:
            return
        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=2.0)
            rospy.ServiceProxy("/gazebo/pause_physics", Empty)()
            self.paused = True
            print("GAZEBO PAUSED", flush=True)
        except Exception as exception:
            print(f"Could not pause Gazebo: {exception}", flush=True)

    def snapshot(self):
        if self.odometry is None:
            return None, "no odometry"
        if time.monotonic() - self.odometry_wall > 2.0:
            return None, "odometry timeout"

        position = self.odometry.pose.pose.position
        velocity = self.odometry.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        values = (position.x, position.y, position.z, speed)

        if not all(math.isfinite(value) for value in values):
            fault = "non-finite state"
        elif (
            abs(position.x) > self.horizontal_limit
            or abs(position.y) > self.horizontal_limit
        ):
            fault = "horizontal boundary exceeded"
        elif position.z < -0.05 or position.z > self.altitude_limit:
            fault = "altitude boundary exceeded"
        elif speed > self.speed_limit:
            fault = "velocity boundary exceeded"
        else:
            fault = None

        return {
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "speed": speed,
        }, fault

    def _wait_for_initial_state(self):
        print("Waiting for fresh Gazebo odometry...", flush=True)
        deadline = time.monotonic() + 30.0
        while self.odometry is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.odometry is None:
            raise RuntimeError("Gazebo odometry unavailable")

        initial, fault = self.snapshot()
        if fault:
            raise RuntimeError(f"Initial-state safety fault: {fault}")
        print(
            "INITIAL STATE "
            f"x={initial['x']:+.6f} y={initial['y']:+.6f} "
            f"z={initial['z']:+.6f} speed={initial['speed']:.6f}",
            flush=True,
        )
        if (
            abs(initial["x"]) > 0.02
            or abs(initial["y"]) > 0.02
            or not 0.03 <= initial["z"] <= 0.10
            or initial["speed"] > 0.05
        ):
            raise RuntimeError("Gazebo is not in the required fresh initial state")

        deadline = time.monotonic() + 10.0
        while self.arm_publisher.get_num_connections() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("RotorS arm subscriber unavailable")
            time.sleep(0.1)
        if not self.lock_and_confirm():
            raise RuntimeError("Unable to confirm zero motor speed before startup")

    def _wait_for_controller(self):
        deadline = time.monotonic() + 120.0
        while not (self.busy_received and not self.busy):
            if self.controller.returncode is not None:
                raise RuntimeError("Controller exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("Controller startup timeout")
            _, fault = self.snapshot()
            if fault:
                raise RuntimeError(f"Startup safety fault: {fault}")
            time.sleep(0.2)
        print("CONTROLLER READY", flush=True)

    def _wait_for_acceptance(self):
        deadline = time.monotonic() + 45.0
        while not self.accepted:
            if self.controller.returncode is not None:
                raise RuntimeError("Controller exited before trajectory acceptance")
            if self.reference.returncode is not None:
                raise RuntimeError("Reference generator exited before acceptance")
            if time.monotonic() > deadline:
                raise RuntimeError("Trajectory acceptance timeout")
            _, fault = self.snapshot()
            if fault:
                raise RuntimeError(f"Pre-flight safety fault: {fault}")
            time.sleep(0.1)
        self.accept_window = False
        print("TRAJECTORY ACCEPTED - FLIGHT STARTED", flush=True)

    def _monitor_flight(self):
        flight_start = time.monotonic()
        last_print = 0.0
        airborne = False

        while not rospy.is_shutdown():
            now = time.monotonic()
            current, fault = self.snapshot()
            if self.controller.returncode is not None:
                fault = "controller exited during flight"
            if self.reference.returncode not in (None, 0):
                fault = "reference generator exited with error"

            if current is not None:
                airborne = airborne or current["z"] >= self.airborne_threshold
                if now - last_print >= 2.0:
                    print(
                        "FLIGHT "
                        f"x={current['x']:+.3f} y={current['y']:+.3f} "
                        f"z={current['z']:+.3f} speed={current['speed']:.3f} "
                        f"busy={self.busy} airborne={airborne}",
                        flush=True,
                    )
                    last_print = now

            if fault:
                raise RuntimeError(fault)
            if (
                airborne
                and current is not None
                and current["z"] < 0.18
                and self.busy_received
                and not self.busy
            ):
                return
            if now - flight_start > self.flight_timeout:
                raise RuntimeError("flight timeout")
            time.sleep(0.1)

        raise RuntimeError("ROS shutdown")

    def run(self):
        status = "UNKNOWN"
        error = ""
        completed = False
        final_lock_confirmed = False

        try:
            self._wait_for_initial_state()
            print("Starting controller...", flush=True)
            self.controller = ManagedProcess(
                "controller", self.controller_command, self.controller_log
            )
            self._wait_for_controller()

            self.arm()
            time.sleep(1.0)
            self.accepted = False
            self.accept_window = True
            print("Starting reference generator...", flush=True)
            self.reference = ManagedProcess(
                "reference generator", self.reference_command, self.reference_log
            )
            self._wait_for_acceptance()
            self._monitor_flight()

            status = "FLIGHT COMPLETE"
            completed = True
            print("FLIGHT COMPLETE", flush=True)
        except KeyboardInterrupt:
            print("FLIGHT INTERRUPTED", flush=True)
            raise
        except Exception as exception:
            error = str(exception)
            status = f"SAFETY FAULT: {error}"
            print(status, flush=True)
        finally:
            pre_stop_lock = self.lock_and_confirm()
            if not completed or not pre_stop_lock:
                self.pause_gazebo()
            cleanup_ok = True
            for process in (self.reference, self.controller):
                if process is None:
                    continue
                try:
                    process.stop()
                except Exception as exception:
                    cleanup_ok = False
                    print(
                        f"PROCESS CLEANUP FAILED ({process.name}): {exception}",
                        flush=True,
                    )
                    self.pause_gazebo()
            if self.paused:
                try:
                    self._publish_arm(False)
                    post_stop_lock = pre_stop_lock
                except Exception as exception:
                    post_stop_lock = False
                    print(
                        f"FINAL LOCK PUBLISH FAILED: {exception}",
                        flush=True,
                    )
            else:
                post_stop_lock = self.lock_and_confirm()
            final_lock_confirmed = pre_stop_lock and post_stop_lock and cleanup_ok
            if not final_lock_confirmed:
                self.pause_gazebo()

        if completed and not final_lock_confirmed:
            completed = False
            error = "final motor lock could not be confirmed"
            status = f"SAFETY FAULT: {error}"

        return FlightOutcome(
            status=status,
            completed=completed,
            paused=self.paused,
            final_lock_confirmed=final_lock_confirmed,
            error=error,
        )


def run_flight(**kwargs):
    return FlightRunner(**kwargs).run()
