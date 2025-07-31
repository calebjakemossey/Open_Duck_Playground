import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default="mujoco_torques.pkl")
args = parser.parse_args()

torques = pickle.load(open(args.file, "rb"))

left_knee_torques = [t[0] for t in torques]
right_knee_torques = [t[1] for t in torques]   
left_ankle_pitch_torques = [t[2] for t in torques]
right_ankle_pitch_torques = [t[3] for t in torques]
plt.figure(figsize=(10, 5))
plt.plot(left_knee_torques, label="Left Knee Torque", color='blue')
plt.plot(right_knee_torques, label="Right Knee Torque", color='orange')
# plt.plot(left_ankle_pitch_torques, label="Left Ankle Pitch Torque", color='green')
# plt.plot(right_ankle_pitch_torques, label="Right Ankle Pitch Torque", color='red')
plt.title("Knee and Ankle Torques Over Time")
plt.xlabel("Time Step")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.close()