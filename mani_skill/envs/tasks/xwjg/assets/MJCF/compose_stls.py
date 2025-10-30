import os
import argparse
from pathlib import Path

def generate_mjcf_from_stl_folder(stl_folder_path, output_xml_path="composite_model.xml"):
    """
    将指定文件夹下的所有 STL 文件生成一个 MuJoCo MJCF (XML) 模型文件。
    所有 STL 文件将被包含在同一个 <body> 中。

    Args:
        stl_folder_path (str): 包含 STL 文件的文件夹路径。
        output_xml_path (str): 输出的 XML 文件路径。
    """
    stl_folder = Path(stl_folder_path)
    if not stl_folder.exists() or not stl_folder.is_dir():
        raise ValueError(f"Path '{stl_folder_path}' is not a valid directory.")

    # 查找所有的 STL 文件
    stl_files = list(stl_folder.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in directory '{stl_folder_path}'.")
        return

    # 开始构建 XML 内容
    model_name = Path(output_xml_path).stem
    xml_content = [f'<mujoco model="{model_name}">']
    xml_content.append('  <compiler angle="radian" meshdir="." />')  # meshdir 设置为 STL 文件夹的父目录
    xml_content.append('''  <default>
        <geom type="mesh"/>
        <default class="material-aluminum">
        <geom density="2700" rgba="0.79216 0.81961 0.93333 1"/>
        </default>
        <default class="material-iron">
        <geom density="7874" rgba="0.3843 0.3569 0.3412 1"/>
        </default>
        <default class="material-mixed">
        <geom rgba="0.7 0.6 0.6 1"/>
        </default>
    </default>''')
    xml_content.append('  <asset>')

    # 为每个 STL 文件添加 <mesh> 元素
    mesh_names = []
    for stl_file in stl_files:
        mesh_name = stl_file.stem  # 使用文件名（不含扩展名）作为 mesh name
        mesh_names.append(mesh_name)
        # mesh 的 file 路径是相对于 meshdir 的。如果 meshdir=".", 则 file="meshes/xxx.stl"
        relative_path = stl_folder.name + "/" + stl_file.name
        xml_content.append(f'    <mesh name="{mesh_name}" file="{relative_path}" scale="1 1 1"/>')

    xml_content.append('  </asset>')
    xml_content.append('  <worldbody>')

    # 创建主要的 body，并将所有 mesh 作为 geom 添加进去
    xml_content.append('    <body name="composite_body" pos="0 0 0">')

    for mesh_name in mesh_names:
        # 这里假设每个 geom 的位置和姿态都已经在 STL 文件中正确设置。
        # 如果部件位置不对，你需要在这里为每个 geom 设置正确的 pos 和 euler 或 quat。
        xml_content.append(f'      <geom type="mesh" mesh="{mesh_name}" pos="0 0 0" quat="1 0 0 0" solref="0.02 1" solimp="0.8 0.8 0.001" class="material-aluminum"/>')

    xml_content.append('        <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>')
    xml_content.append('    </body>')
    xml_content.append('  </worldbody>')
    xml_content.append('</mujoco>')

    # 将生成的 XML 内容写入文件
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(xml_content))

    print(f"MJCF model has been generated and saved to: {output_xml_path}")
    print(f"Number of STL files processed: {len(stl_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a MuJoCo MJCF model from a folder of STL files.')
    parser.add_argument('stl_folder', type=str, help='Path to the folder containing STL files.')
    parser.add_argument('--output', '-o', type=str, default="composite_model.xml", help='Output path for the generated XML file.')

    args = parser.parse_args()

    generate_mjcf_from_stl_folder(args.stl_folder, args.output)