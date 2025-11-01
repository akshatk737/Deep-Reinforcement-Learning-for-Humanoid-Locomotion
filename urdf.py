#module2
import math
import textwrap
from pathlib import Path

def inertia_box(m, x, y, z):
    Ixx = (1/12) * m * (y*y + z*z)
    Iyy = (1/12) * m * (x*x + z*z)
    Izz = (1/12) * m * (x*x + y*y)
    return Ixx, Iyy, Izz

def inertia_cylinder_z(m, r, h):

    Ixx = Iyy = (1/12) * m * (3*r*r + h*h)
    Izz = 0.5 * m * r*r
    return Ixx, Iyy, Izz



def make_humanoid_urdf(file_path="humanoid.urdf"):

    torso = dict(name="torso", mass=10.0, size=(0.25, 0.15, 0.40))
    head = dict(name="head", mass=3.0, radius=0.10, height=0.12)
    upper_arm = dict(mass=1.5, length=0.28, radius=0.05)
    lower_arm = dict(mass=1.0, length=0.25, radius=0.045)
    upper_leg = dict(mass=3.0, length=0.40, radius=0.06)
    lower_leg = dict(mass=2.5, length=0.40, radius=0.06)

    tx, ty, tz = torso['size']
    It = inertia_box(torso['mass'], tx, ty, tz)
    Ih_head = inertia_cylinder_z(head['mass'], head['radius'], head['height'])

    urdf = []
    urdf.append('<?xml version="1.0"?>')
    urdf.append('<robot name="simple_humanoid">')

    urdf.append(f'''
    <link name="{torso['name']}">
      <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="{torso['mass']}"/>
        <inertia ixx="{It[0]:.6f}" ixy="0" ixz="0" iyy="{It[1]:.6f}" iyz="0" izz="{It[2]:.6f}"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <box size="{tx} {ty} {tz}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <box size="{tx} {ty} {tz}"/>
        </geometry>
      </collision>
    </link>
    ''')

    urdf.append(f'''
    <link name="head">
      <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="{head['mass']}"/>
        <inertia ixx="{Ih_head[0]:.6f}" ixy="0" ixz="0" iyy="{Ih_head[1]:.6f}" iyz="0" izz="{Ih_head[2]:.6f}"/>
      </inertial>
      <visual>
        <origin xyz="0 0 {head['height']/2:.3f}" rpy="0 0 0" />
        <geometry>
          <cylinder length="{head['height']}" radius="{head['radius']}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 {head['height']/2:.3f}" rpy="0 0 0" />
        <geometry>
          <cylinder length="{head['height']}" radius="{head['radius']}"/>
        </geometry>
      </collision>
    </link>
    ''')

    def limb_link(name, mass, length, radius, parent_link, joint_name, joint_axis, joint_origin_xyz, lower, upper, effort, velocity, child_offset_xyz=(0,0,0)):
        I = inertia_cylinder_z(mass, radius, length)
        link_block = f'''
        <link name="{name}">
          <inertial>
            <origin xyz="0 0 {length/2:.6f}" rpy="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx="{I[0]:.6f}" ixy="0" ixz="0" iyy="{I[1]:.6f}" iyz="0" izz="{I[2]:.6f}"/>
          </inertial>
          <visual>
            <origin xyz="0 0 {length/2:.6f}" rpy="0 0 0" />
            <geometry>
              <cylinder length="{length}" radius="{radius}"/>
            </geometry>
          </visual>
          <collision>
            <origin xyz="0 0 {length/2:.6f}" rpy="0 0 0" />
            <geometry>
              <cylinder length="{length}" radius="{radius}"/>
            </geometry>
          </collision>
        </link>
        '''
        joint_block = f'''
        <joint name="{joint_name}" type="revolute">
          <parent link="{parent_link}"/>
          <child link="{name}"/>
          <origin xyz="{joint_origin_xyz[0]} {joint_origin_xyz[1]} {joint_origin_xyz[2]}" rpy="0 0 0"/>
          <axis xyz="{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"/>
          <limit effort="{effort}" velocity="{velocity}" lower="{lower}" upper="{upper}"/>
        </joint>
        <transmission type="simple_transmission">
          <actuator name="{joint_name}_motor"/>
          <joint name="{joint_name}"/>
          <mechanicalReduction>1</mechanicalReduction>
        </transmission>
        '''
        return link_block + joint_block

    head_joint = '''
    <joint name="neck_yaw" type="revolute">
      <parent link="torso"/>
      <child link="head"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit effort="10.0" velocity="1.5" lower="-1.57" upper="1.57"/>
    </joint>
    <transmission type="simple_transmission">
      <actuator name="neck_yaw_motor"/>
      <joint name="neck_yaw"/>
      <mechanicalReduction>1</mechanicalReduction>
    </transmission>
    '''
    urdf.append(head_joint)

    urdf.append(limb_link(
        name="left_upper_arm",
        mass=upper_arm['mass'],
        length=upper_arm['length'],
        radius=upper_arm['radius'],
        parent_link="torso",
        joint_name="left_shoulder_pitch",
        joint_axis=(0,1,0),
        joint_origin_xyz=(0.15, 0.18, 0.15),
        lower="-2.0", upper="1.0", effort=50.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="left_lower_arm",
        mass=lower_arm['mass'],
        length=lower_arm['length'],
        radius=lower_arm['radius'],
        parent_link="left_upper_arm",
        joint_name="left_elbow",
        joint_axis=(0,1,0),
        joint_origin_xyz=(0,0,upper_arm['length']),
        lower="-2.0", upper="0.0", effort=30.0, velocity=4.0
    ))

    # Right arm (mirror on Y)
    urdf.append(limb_link(
        name="right_upper_arm",
        mass=upper_arm['mass'],
        length=upper_arm['length'],
        radius=upper_arm['radius'],
        parent_link="torso",
        joint_name="right_shoulder_pitch",
        joint_axis=(0,1,0),
        joint_origin_xyz=(0.15, -0.18, 0.15),
        lower="-2.0", upper="1.0", effort=50.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="right_lower_arm",
        mass=lower_arm['mass'],
        length=lower_arm['length'],
        radius=lower_arm['radius'],
        parent_link="right_upper_arm",
        joint_name="right_elbow",
        joint_axis=(0,1,0),
        joint_origin_xyz=(0,0,upper_arm['length']),
        lower="-2.0", upper="0.0", effort=30.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="left_upper_leg",
        mass=upper_leg['mass'],
        length=upper_leg['length'],
        radius=upper_leg['radius'],
        parent_link="torso",
        joint_name="left_hip",
        joint_axis=(1,0,0),
        joint_origin_xyz=(0.0, 0.08, -0.2),
        lower="-1.57", upper="1.0", effort=100.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="left_lower_leg",
        mass=lower_leg['mass'],
        length=lower_leg['length'],
        radius=lower_leg['radius'],
        parent_link="left_upper_leg",
        joint_name="left_knee",
        joint_axis=(1,0,0),
        joint_origin_xyz=(0,0,upper_leg['length']),
        lower="0.0", upper="2.0", effort=80.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="right_upper_leg",
        mass=upper_leg['mass'],
        length=upper_leg['length'],
        radius=upper_leg['radius'],
        parent_link="torso",
        joint_name="right_hip",
        joint_axis=(1,0,0),
        joint_origin_xyz=(0.0, -0.08, -0.2),
        lower="-1.57", upper="1.0", effort=100.0, velocity=4.0
    ))

    urdf.append(limb_link(
        name="right_lower_leg",
        mass=lower_leg['mass'],
        length=lower_leg['length'],
        radius=lower_leg['radius'],
        parent_link="right_upper_leg",
        joint_name="right_knee",
        joint_axis=(1,0,0),
        joint_origin_xyz=(0,0,upper_leg['length']),
        lower="0.0", upper="2.0", effort=80.0, velocity=4.0
    ))

    urdf.append('</robot>')

    content = "\n".join(urdf)

    p = Path(file_path)
    p.write_text(content)
    print(f"URDF written to {p.resolve()}")
    return p.resolve(), content

urdf_path, urdf_content = make_humanoid_urdf("humanoid.urdf")
print("Sample URDF length:", len(urdf_content))

print(urdf_content[:800])


import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.8)
plane = p.loadURDF("plane.urdf")
humanoid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1])
p.setRealTimeSimulation(0)
while True:
    p.stepSimulation()
    time.sleep(1./240.)

