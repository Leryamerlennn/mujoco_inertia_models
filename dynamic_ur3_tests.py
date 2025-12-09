import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


XML_MODEL = "robot_with_CRA-RI60-80.xml"  

DT = 0.002  
T_JS = 4.0  
T_CART = 8.0  

TAU_MAX = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])
OMEGA_MAX = np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])



def smooth_cos_trajectory(q0, qf, T, dt):

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    q = np.zeros((N, len(q0)))
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)

    dq = qf - q0

    for i, ti in enumerate(t):
        s = 0.5 * (1 - np.cos(np.pi * ti / T))
        sdot = 0.5 * (np.pi / T) * np.sin(np.pi * ti / T)
        sddot = 0.5 * (np.pi / T)**2 * np.cos(np.pi * ti / T)

        q[i] = q0 + dq * s
        qdot[i] = dq * sdot
        qddot[i] = dq * sddot

    return t, q, qdot, qddot


def resolved_rate_cartesian_trajectory(model, data, q0, T, dt, radius=0.10, K=1.0):

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)

    nq = model.nq
    nv = model.nv

    q = np.zeros((N, nq))
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)

    q[0] = q0
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    ee_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
    p0 = data.xpos[ee_body].copy()

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for i in range(N - 1):
        ti = t[i]

        angle = 2 * np.pi * ti / T
        pd = np.array([
            p0[0] + radius * np.cos(angle),
            p0[1] + radius * np.sin(angle),
            p0[2]
        ])

        data.qpos[:] = q[i]
        data.qvel[:] = qdot[i]
        mj.mj_forward(model, data)

        p = data.xpos[ee_body].copy()
        e = pd - p  

        jacp[:] = 0.0
        jacr[:] = 0.0
        mj.mj_jac(model, data, jacp, jacr, p, ee_body)

        J = jacp[:, :nq] 

        J_pinv = np.linalg.pinv(J)

        v_des = K * e  
        qdot_cmd = J_pinv @ v_des

        for j in range(nq):
            w_max = OMEGA_MAX[j]
            qdot_cmd[j] = np.clip(qdot_cmd[j], -w_max * 0.8, w_max * 0.8)

        qdot[i+1] = qdot_cmd
        q[i+1] = q[i] + qdot_cmd * dt

    for i in range(1, N-1):
        qddot[i] = (qdot[i+1] - qdot[i-1]) / (2 * dt)
    qddot[0] = (qdot[1] - qdot[0]) / dt
    qddot[-1] = (qdot[-1] - qdot[-2]) / dt

    return t, q, qdot, qddot


def compute_torque_along_trajectory(model, data, q, qdot, qddot):

    N, nq = q.shape
    tau = np.zeros((N, nq))

    for i in range(N):
        data.qpos[:] = q[i]
        data.qvel[:] = qdot[i]
        data.qacc[:] = qddot[i]
        mj.mj_forward(model, data)
        mj.mj_inverse(model, data)
        tau[i] = data.qfrc_inverse[:nq]

    return tau

def main():
    base_dir = Path(__file__).parent
    xml_path = base_dir / XML_MODEL

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    nq = model.nq

    q0_js = np.zeros(nq)
    qf_js = np.array([np.pi/3, -np.pi/4, np.pi/6, 0.0, 0.0, 0.0])

    t_js, q_js, qd_js, qdd_js = smooth_cos_trajectory(q0_js, qf_js, T_JS, DT)
    tau_js = compute_torque_along_trajectory(model, data, q_js, qd_js, qdd_js)

    q0_cart = np.array([0.0, -np.pi/3, 2*np.pi/3, 0.0, np.pi/6, 0.0])
    t_cart, q_cart, qd_cart, qdd_cart = resolved_rate_cartesian_trajectory(
        model, data, q0_cart, T_CART, DT, radius=0.10, K=1.0
    )
    tau_cart = compute_torque_along_trajectory(model, data, q_cart, qd_cart, qdd_cart)

   
    joint_index = 1  #  J2

    omega_js = qd_js[:, joint_index]
    tau_js_j = tau_js[:, joint_index]

    omega_cart = qd_cart[:, joint_index]
    tau_cart_j = tau_cart[:, joint_index]

    tau_max = TAU_MAX[joint_index]
    omega_max = OMEGA_MAX[joint_index]

    plt.figure(figsize=(7, 6))
    plt.scatter(omega_js, tau_js_j, s=10, alpha=0.6, label="Joint-space trajectory")
    plt.scatter(omega_cart, tau_cart_j, s=10, alpha=0.6, label="Cartesian trajectory")

    plt.axvline(x= omega_max, linestyle="--", color="k", label="ω_max")
    plt.axvline(x=-omega_max, linestyle="--", color="k")
    plt.axhline(y= tau_max, linestyle="--", color="r", label="τ_max")
    plt.axhline(y=-tau_max, linestyle="--", color="r")

    plt.xlabel("Joint angular velocity ω [rad/s]")
    plt.ylabel("Joint torque τ [Nm]")
    plt.title(f"Joint {joint_index+1}: torque-speed operating region")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Max |ω| for joint {joint_index+1} (JS): {np.max(np.abs(omega_js)):.3f} rad/s")
    print(f"Max |ω| for joint {joint_index+1} (Cart): {np.max(np.abs(omega_cart)):.3f} rad/s")
    print(f"Limit ω_max: {omega_max:.3f} rad/s")

    print(f"Max |τ| for joint {joint_index+1} (JS): {np.max(np.abs(tau_js_j)):.3f} Nm")
    print(f"Max |τ| for joint {joint_index+1} (Cart): {np.max(np.abs(tau_cart_j)):.3f} Nm")
    print(f"Limit τ_max: {tau_max:.3f} Nm")


if __name__ == "__main__":
    main()
