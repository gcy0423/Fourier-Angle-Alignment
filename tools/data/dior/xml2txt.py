import os
import xml.etree.ElementTree as ET

# 输入文件夹路径和输出文件夹路径
input_folder = '/data2/guchangyu/datasets/DIOR-R/Annotations/Oriented Bounding Boxes'
output_folder = '/data2/guchangyu/datasets/DIOR-R/Annotations/Oriented Bounding Boxes TXT'
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有XML文件
for filename in os.listdir(input_folder):
    if filename.endswith('.xml'):
        xml_path = os.path.join(input_folder, filename)
        print(f'convert {filename}')

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 创建用于存储文本数据的列表
        text_data = []

        # 遍历XML元素并提取所需的信息
        for obj in root.findall('.//object'):
            robndbox = obj.find('robndbox')
            if robndbox is not None:
                x_left_top = robndbox.find('x_left_top').text
                y_left_top = robndbox.find('y_left_top').text
                x_right_top = robndbox.find('x_right_top').text
                y_right_top = robndbox.find('y_right_top').text
                x_right_bottom = robndbox.find('x_right_bottom').text
                y_right_bottom = robndbox.find('y_right_bottom').text
                x_left_bottom = robndbox.find('x_left_bottom').text
                y_left_bottom = robndbox.find('y_left_bottom').text
                name = obj.find('name').text
                difficult = obj.find('difficult').text

                # 将提取的信息格式化并添加到文本数据列表中
                text_line = f"{x_left_top} {y_left_top} {x_right_top} {y_right_top} {x_right_bottom} {y_right_bottom} {x_left_bottom} {y_left_bottom} {name} {difficult}\n"
                text_data.append(text_line)

        # 创建输出文件路径
        output_path = os.path.join(output_folder, filename.replace('.xml', '.txt'))

        # 将文本数据写入输出文件
        with open(output_path, 'w') as output_file:
            output_file.writelines(text_data)

print("转换完成！")
