import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_FILES = [
    "robot_no_added_inertia.xml",
    "robot_with_added_inertia.xml",
    "robot_with_CRA-RI30-40.xml",
    "robot_with_CRA-RI40-52.xml",
    "robot_with_CRA-RI50-70.xml",
    "robot_with_CRA-RI60-80.xml",
    "robot_with_CRA-RI70-90.xml",
    "robot_with_CRA-RI80-110.xml",
]

MOTOR_RATED_TORQUE = {
    "robot_no_added_inertia.xml":   50.0, 
    "robot_with_added_inertia.xml": 59.0,  
    "robot_with_CRA-RI30-40.xml":   59.0,
    "robot_with_CRA-RI40-52.xml":   78.0,
    "robot_with_CRA-RI50-70.xml":   98.0,
    "robot_with_CRA-RI60-80.xml":   196.0,
    "robot_with_CRA-RI70-90.xml":   245.0,
    "robot_with_CRA-RI80-110.xml":  343.0,
}


NUM_SAMPLES = 100  

def sample_static_configs(model, num_samples=NUM_SAMPLES):
    nq = model.nq
    qpos_samples = []

    
    jnt_range = model.jnt_range.copy()

    for _ in range(num_samples):
        q = np.zeros(nq)
        for j in range(nq):
            low, high = jnt_range[j]
            if low == 0 and high == 0:
                low, high = -2*np.pi, 2*np.pi
            q[j] = np.random.uniform(low, high)
        qpos_samples.append(q)

    return np.array(qpos_samples)

def run_static_experiment(xml_path):
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    qpos_samples = sample_static_configs(model)
    nq = model.nq

    torques = []

    for q in qpos_samples:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

        mj.mj_forward(model, data)
        mj.mj_inverse(model, data)

        tau = data.qfrc_inverse[:nq].copy()   
        torques.append(tau)

    torques = np.vstack(torques)       
    max_abs_tau = np.max(np.abs(torques), axis=0)  

    return torques, max_abs_tau

def main():
    base_dir = Path(__file__).parent

    all_results = {}        
    safe_models = []     

    for xml_name in MODEL_FILES:
        xml_path = base_dir / xml_name
        if not xml_path.exists():
            print(f"WARNING: {xml_path} not found, skipping")
            continue

        torques, max_abs_tau = run_static_experiment(xml_path)
        all_results[xml_name] = (torques, max_abs_tau)

        rated = MOTOR_RATED_TORQUE[xml_name]
        if np.all(max_abs_tau <= rated):
            safe_models.append(xml_name)

        print(f"Model: {xml_name}")
        print(f"  Max |tau| per joint: {max_abs_tau}")
        print(f"  Rated torque: {rated} N·m")
        print(f"  OK for all joints? {np.all(max_abs_tau <= rated)}")
        print()

    plt.figure(figsize=(12, 6))
    joint_indices = None

    for i, (xml_name, (_, max_abs_tau)) in enumerate(all_results.items()):
        nq = len(max_abs_tau)
        if joint_indices is None:
            joint_indices = np.arange(nq)
        offset = (i - len(all_results)/2)*0.1
        plt.bar(joint_indices + offset, max_abs_tau,
                width=0.1, label=xml_name)

    plt.xlabel("Joint index")
    plt.ylabel("Max |torque|, N·m")
    plt.title("Static max joint torques for different models")
    plt.legend(fontsize=6)
    plt.grid(True)

    joint_to_plot = 1  # J2
    plt.figure(figsize=(8, 4))
    data_for_violin = []
    labels = []
    for xml_name, (torques, _) in all_results.items():
        data_for_violin.append(np.abs(torques[:, joint_to_plot]))
        labels.append(xml_name)

    plt.violinplot(data_for_violin, showmeans=True, showextrema=True)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=60, ha="right")
    plt.ylabel(f"|tau| at joint {joint_to_plot} (N·m)")
    plt.title("Static torque distribution at joint J2")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Models where all joints are below rated torque:")
    for m in safe_models:
        print("  -", m)


if __name__ == "__main__":
    main()
