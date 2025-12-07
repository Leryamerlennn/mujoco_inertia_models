from lxml import etree
import copy
import math

MOTOR_CATALOG = {
    "CRA-RI80-110": {"J": 1255, "ratio": 101, "torque": 343},
    "CRA-RI70-90": {"J": 592,  "ratio": 101, "torque": 245},
    "CRA-RI60-80": {"J": 441,  "ratio": 101, "torque": 196},
    "CRA-RI50-70": {"J": 134,  "ratio": 101, "torque": 98},
    "CRA-RI40-52": {"J": 80,   "ratio": 101, "torque": 78},
    "CRA-RI30-40": {"J": 24.5, "ratio": 101, "torque": 59},
}

JOINTS = ["j1", "j2", "j3"]   
LINKS = ["link1", "link2", "link3"]

def reflected_inertia(J_gcm2, ratio):
    J = J_gcm2 * 1e-7 
    return J * (ratio ** 2)

def motor_cylinder_inertia(m, R, h):
    Ixy = (1/12) * m * (3*R**2 + h**2)
    Iz = 0.5 * m * R**2
    return Ixy, Ixy, Iz


def parallel_axis(I, m, dx, dy, dz):
    d2 = dx*dx + dy*dy + dz*dz
    return (I + m * d2)



def update_link_inertia(link_elem, motor_mass, I_motor_diag, com_offset):


    inertial = link_elem.find("inertial")
    mass = float(inertial.get("mass"))
    Ixx = float(inertial.get("ixx"))
    Iyy = float(inertial.get("iyy"))
    Izz = float(inertial.get("izz"))

    dx, dy, dz = com_offset

    new_mass = mass + motor_mass
    inertial.set("mass", f"{new_mass}")

    I_mx = parallel_axis(I_motor_diag[0], motor_mass, dx, dy, dz)
    I_my = parallel_axis(I_motor_diag[1], motor_mass, dx, dy, dz)
    I_mz = parallel_axis(I_motor_diag[2], motor_mass, dx, dy, dz)

    inertial.set("ixx", f"{Ixx + I_mx}")
    inertial.set("iyy", f"{Iyy + I_my}")
    inertial.set("izz", f"{Izz + I_mz}")


def set_joint_armature(root, joint_name, armature_value):
    for j in root.findall(".//joint"):
        if j.get("name") == joint_name:
            j.set("armature", str(armature_value))


def ensure_actuators_and_sensors(root):
    actuator = root.find("actuator")
    if actuator is None:
        actuator = etree.SubElement(root, "actuator")

    for j in JOINTS:
        etree.SubElement(
            actuator,
            "position",
            name=f"{j}_motor",
            joint=j,
            kp="100",
            ctrlrange="-1 1"
        )

    sensors = root.find("sensor")
    if sensors is None:
        sensors = etree.SubElement(root, "sensor")

    for j in JOINTS:
        etree.SubElement(
            sensors,
            "jointpos",
            name=f"{j}_pos",
            joint=j
        )


def generate_models(base_xml_path):
    tree = etree.parse(base_xml_path)
    root = tree.getroot()

    for motor_name, props in MOTOR_CATALOG.items():
        J_ref = reflected_inertia(props["J"], props["ratio"])
        torque_limit = props["torque"]

        new_root = copy.deepcopy(root)

        for joint in JOINTS:
            set_joint_armature(new_root, joint, J_ref)

        ensure_actuators_and_sensors(new_root)

        outfile = f"robot_with_{motor_name}.xml"
        new_tree = etree.ElementTree(new_root)
        new_tree.write(outfile, pretty_print=True, encoding="utf-8", xml_declaration=True)
        print("Written", outfile)


if __name__ == "__main__":
    generate_models("universalUR3_mjcf.xml")
