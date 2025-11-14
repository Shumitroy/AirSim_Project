

from __future__ import annotations

import time

import airsim  

from config import CONFIG


def _connect_client() -> airsim.MultirotorClient:
    client = airsim.MultirotorClient(ip=CONFIG.ip, port=CONFIG.port)
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=CONFIG.vehicle_name)
    client.armDisarm(True, vehicle_name=CONFIG.vehicle_name)
    return client


def run_stable_velocity_demo() -> None:
    client = _connect_client()
    print("[Stable Velocity] Connected to AirSim.")

    print("Taking off...")
    client.takeoffAsync(vehicle_name=CONFIG.vehicle_name).join()
    client.moveToZAsync(
        CONFIG.default_altitude,
        CONFIG.default_velocity,
        vehicle_name=CONFIG.vehicle_name,
    ).join()

    print("Flying forward with constant velocity...")
    duration = 5.0  # seconds
    vx = CONFIG.default_velocity
    vy = 0.0
    vz = 0.0

    client.moveByVelocityAsync(
        vx,
        vy,
        vz,
        duration=duration,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        vehicle_name=CONFIG.vehicle_name,
    ).join()

    print("Hovering...")
    client.hoverAsync(vehicle_name=CONFIG.vehicle_name).join()
    time.sleep(2.0)

    print("Landing...")
    client.landAsync(vehicle_name=CONFIG.vehicle_name).join()
    client.armDisarm(False, vehicle_name=CONFIG.vehicle_name)
    client.enableApiControl(False, vehicle_name=CONFIG.vehicle_name)
    print("Done.")
