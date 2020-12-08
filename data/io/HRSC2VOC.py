# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:12
# @Author  : Peng Miao
# @File    : HRSC2VOC.py
# @Intro   : 【适用于HRSC2016数据集标注文件】将HRSC2016标注文件转换为适合该模型的VOC格式标注文件，并且从5点标注转换为8点标注
#            'mbox_ang'标签代表的是旋转角度（与x轴负轴的夹角大小，范围在-pi/2 ~ pi/2之间）


import os
import xml.etree.ElementTree as ET
import math
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import glob

# origin_ann_dir = 'voc_ship/JPEGImages/'
# new_ann_dir = 'text/'

XML_ROOT_FOLDER = 'G:\毕设数据集\HRSC2016\HRSC2016\FullDataSet\Annotations'
XML_DES_FOLDER = 'D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN-DFPN_RPN_HEAD_AROI\data\VOCdevkit\Annotations'
pi = 3.141592


# 解析文件名出来
def GetXMLFiles():
    global XML_ROOT_FOLDER
    file_list = []
    files = os.listdir(XML_ROOT_FOLDER)
    for file in files:
        if file.endswith(".xml"):
            # shutil.move(os.path.join(DATA_ROOT_FOLDER, file), os.path.join(DATA_ROOT_FOLDER, file.replace('.tif', '.tiff')))
            file_list.append(file)
    return file_list


xml_Lists = GetXMLFiles()
len(xml_Lists)

xml_basenames = []  # e.g. 100.jpg
for item in xml_Lists:
    xml_basenames.append(os.path.basename(item))

xml_names = []  # e.g. 100
for item in xml_basenames:
    temp1, temp2 = os.path.splitext(item)
    xml_names.append(temp1)

count=0

for it in xml_names:
    tree = ET.parse(os.path.join(XML_ROOT_FOLDER, str(it) + '.xml'))
    root = tree.getroot()

    # 创建Element，组装成新的xml文件
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train_images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(it) + '.bmp'  # str(1) + '.jpg'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = root.find('Img_SizeWidth').text  # 在原xml文件中查找width
    node_height = SubElement(node_size, 'height')
    node_height.text = root.find('Img_SizeHeight').text  # 在原xml文件中查找height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = root.find('Img_SizeDepth').text  # 在原xml文件中查找depth

    for Object in root.findall('./HRSC_Objects/HRSC_Object'):
        mbox_cx = float(Object.find('mbox_cx').text)
        mbox_cy = float(Object.find('mbox_cy').text)
        mbox_w = float(Object.find('mbox_w').text)
        mbox_h = float(Object.find('mbox_h').text)
        mbox_ang = float(Object.find('mbox_ang').text)
        print("原始坐标：[{},{},{},{},{}]".format(mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang))

        # 计算舰首与舰尾点的坐标

        bow_x = mbox_cx + mbox_w / 2 * math.cos(mbox_ang)
        bow_y = mbox_cy + mbox_w / 2 * math.sin(mbox_ang)

        tail_x = mbox_cx - mbox_w / 2 * math.cos(mbox_ang)
        tail_y = mbox_cy - mbox_w / 2 * math.sin(mbox_ang)

        print('舰首舰尾坐标：[{},{},{},{}]'.format(bow_x,bow_y,tail_x,tail_y))

        # 根据舰首舰尾的坐标，结合宽高，计算旋转矩形框四个顶点的坐标

        bowA_x = round(bow_x + mbox_h / 2 * math.sin(mbox_ang))
        bowA_y = round(bow_y - mbox_h / 2 * math.cos(mbox_ang))

        bowB_x = round(bow_x - mbox_h / 2 * math.sin(mbox_ang))
        bowB_y = round(bow_y + mbox_h / 2 * math.cos(mbox_ang))

        tailA_x = round(tail_x + mbox_h / 2 * math.sin(mbox_ang))
        tailA_y = round(tail_y - mbox_h / 2 * math.cos(mbox_ang))

        tailB_x = round(tail_x - mbox_h / 2 * math.sin(mbox_ang))
        tailB_y = round(tail_y + mbox_h / 2 * math.cos(mbox_ang))

        print("转换后的坐标：[{},{},{},{},{},{},{},{}]".format(bowA_x, bowA_y, bowB_x, bowB_y, tailA_x, tailA_y, tailB_x, tailB_y))
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'text'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')

        node_x1 = SubElement(node_bndbox, 'x1')
        node_x1.text = str(bowA_x)
        node_y1 = SubElement(node_bndbox, 'y1')
        node_y1.text = str(bowA_y)

        node_x2 = SubElement(node_bndbox, 'x2')
        node_x2.text = str(bowB_x)
        node_y2 = SubElement(node_bndbox, 'y2')
        node_y2.text = str(bowB_y)

        node_x3 = SubElement(node_bndbox, 'x3')
        node_x3.text = str(tailA_x)
        node_y3 = SubElement(node_bndbox, 'y3')
        node_y3.text = str(tailA_y)

        node_x4 = SubElement(node_bndbox, 'x4')
        node_x4.text = str(tailB_x)
        node_y4 = SubElement(node_bndbox, 'y4')
        node_y4.text = str(tailB_y)

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(min(bowA_x, bowB_x, tailA_x, tailB_x))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(min(bowA_y, bowB_y, tailA_y, tailB_y))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(max(bowA_x, bowB_x, tailA_x, tailB_x))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(max(bowA_y, bowB_y, tailA_y, tailB_y))

    # break
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    fw = open(os.path.join(XML_DES_FOLDER, str(it) + '.xml'), 'wb')
    fw.write(xml)
    print("----------------------------------------------------------")
    fw.close()

    # count+=1
    # if(count==2):
    #     break

print("----------------------------------------转换完成----------------------------------------")