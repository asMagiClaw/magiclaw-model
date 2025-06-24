
import argparse
import os
import mujoco
import mujoco.viewer
import numpy as np

parser = argparse.ArgumentParser(description="Run a MuJoCo simulation with a viewer.")
parser.add_argument("--model", type=str, default="magiclaw/mjcf/magiclaw-on-robot-surf.xml", help="Path to the MuJoCo model XML file.")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.model)
data = mujoco.MjData(model)

print("=== All joints ===")
for i in range(model.njnt):
    joint_name = model.joint(i).name
    qposadr = model.joint(i).qposadr
    joint_type = model.joint(i).type
    print(f"[{i}] name={joint_name}, qposadr={qposadr}, type={joint_type}")

data.qpos[:] = np.zeros(model.nq)
data.qvel[:] = np.zeros(model.nv)

data.ctrl[:] = np.zeros(model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()