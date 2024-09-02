from PIL import Image
import os
import json
import math
from numba import jit
import numpy as np
import cv2

def create_data_lists(input_folders,output_folder,split):
    """
    Create COCP-SR dataset.
    Args：
        input_folders: Path to the input folder.
        output_folder: Output path to the json file.
        split: train/test/val.
    """
    print("\nCreating COCP-SR dataset %s file list.\n" % split)
    path_images = list()
    json_name = ''

    for i in os.listdir(input_folders):
        img_path = os.path.join(input_folders, i)
        path_images.append(img_path)

    if split == 'train':
        json_name = 'train_images.json'
    elif split == 'validation':
        json_name = 'val_images.json'
    elif split == 'test':
        json_name = 'test_images.json'

    with open(os.path.join(output_folder, json_name), 'w') as j:
        json.dump(path_images, j)

    print("Generation completed. There are %d images in total.\n" % len(path_images))
    print("The list of image file paths has been saved under %s\n" % output_folder)

def create_DFdata_lists(input_folders, min_size_h, min_size_w, output_folder,split):
    """
    Create SIM-SR dataset.
    Args：
        input_folders: Path to the input folder.
        min_size_h: Minimum height of input image.
        min_size_w: Minimum width of input image.
        output_folder: Output path to the json file.
        split: train/test/val.
    """
    print("\nCreating SIM-SR dataset %s file list.\n" % split)
    output_path = ''
    json_name = ''
    if split == 'train':
        output_path = output_folder + "/DF_train"
        os.makedirs(output_path, exist_ok=True)
        json_name = 'train_images.json'
    elif split == 'validation':
        output_path = output_folder + "/DF_val"
        os.makedirs(output_path, exist_ok=True)
        json_name = 'val_images.json'

    path_images = list()
    idx = 0
    for d in input_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = cv2.imread(img_path, flags=1)
            img_height, img_width, channels = img.shape
            if img_width >= min_size_w and img_height >= min_size_h:
                idx += 1
                filename = f'/{idx: 04d}.png'
                cv2.imwrite(output_path + filename, img)

    for i in os.listdir(output_path):
        img_path = os.path.join(output_path, i)
        path_images.append(img_path)

    with open(os.path.join(output_folder, json_name), 'w') as j:
        json.dump(path_images, j)

    print("Generation completed. There are %d images in total.\n" % len(path_images))
    print("The image file has been saved under %s.\n" % output_path)
    print("The json file has been saved under %s.\n" % output_folder)

def zm_shape(a,c,f,L,dx,dy,r1,r2):
    '''
    Calculate the height h_zm, width w_zm, and Z-axis coordinate h2 of the cylindrical expansion diagram.
    Args：
        a, c:Mirror parameters.
        f:Camera focal length.
        L:Cylindrical radius.
        dx, dy:pixel size.
        r1:Catadioptric image inner circle radius.
        r2:Catadioptric image outer circle radius.
    '''
    r1_x=r1*dx
    r2_x=r2*dx
    #Angle between the reflected light at the lower edge of the cylinder and the Z-axis
    ag_z1 = np.arctan(r1_x/f)
    #Angle between the reflected light at the upper edge of the cylindrical surface and the Z-axis
    ag_z2 = np.arctan(r2_x/f)

    p1 = (c ** 2 + a ** 2 - 2 * a * c * math.cos(ag_z1)) / (c * math.cos(ag_z1) - a)
    tan_angle_xoy1 = ((2 * a + p1) * math.cos(ag_z1) - 2 * c) / ((2 * a + p1) * math.sin(ag_z1))
    p2 = (c ** 2 + a ** 2 - 2 * a * c * math.cos(ag_z2)) / (c * math.cos(ag_z2) - a)
    tan_angle_xoy2 = ((2 * a + p2) * math.cos(ag_z2) - 2 * c) / ((2 * a + p2) * math.sin(ag_z2))

    h1_y = L * tan_angle_xoy1
    h2_y = L * tan_angle_xoy2
    h1 = int(h1_y / dx)
    h2 = int(h2_y / dx)
    print(h2)
    h_zm = h2 - h1
    w_zm = int(2 * math.pi * L / dy)

    return h_zm, w_zm, h2

@jit(nopython=True)
def coefficient(x):
    a = -0.5
    x = abs(x)
    if x <= 1:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    elif 1 < x < 2:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    else:
        return 0

@jit(nopython=True)
def F_maptable(h_qj, w_qj,h2, a,b,c,f,L,dx,dy,m_dot,n_dot,r1,r2):
    '''
    Calculate the coordinate mapping table from catadioptric image to cylindrical unfolded image.
    Args：
        h_qj: Height of Catadioptric image
        w_qj: Width of Catadioptric image
        h2: Distance from the upper edge of the cylindrical surface to the x-axis
        a、b、c: Mirror parameters
        f: Camera focal length
        L: Cylindrical radius
        dx, dy: Pixel size
        m_dot、n_dot: Center coordinates of panoramic image, m rows and n columns
        r1: Catadioptric image inner circle radius
        r2: Catadioptric image outer circle radius
        f_map_table：Forward coordinate mapping table
    '''
    f_map_table = np.zeros((h_qj, w_qj, 2), dtype=np.float64)
    # Calculate coordinate mapping table
    for i in range(h_qj):
        for j in range(w_qj):
            if r1 <= math.sqrt(((m_dot - i) ** 2 + (n_dot - j) ** 2)) <= r2:
                x_qj = (m_dot - i) * dx
                y_qj = (n_dot - j) * dy

                cos_angle_xoz = x_qj / math.sqrt(x_qj ** 2 + y_qj ** 2)
                # sin_angle_xoz = y_qj / math.sqrt(x_qj ** 2 + y_qj ** 2)

                cos_angle_z = f / math.sqrt(x_qj ** 2 + y_qj ** 2 + f ** 2)
                sin_angle_z = math.sqrt(x_qj ** 2 + y_qj ** 2) / math.sqrt(x_qj ** 2 + y_qj ** 2 + f ** 2)

                p = (c ** 2 + a ** 2 - 2 * a * c * cos_angle_z) / (c * cos_angle_z - a)

                tan_angle_xoy = ((2 * a + p) * cos_angle_z - 2 * c) / ((2 * a + p) * sin_angle_z)

                zc = L * tan_angle_xoy

                if y_qj < 0:
                    angle_xoz = 2 * math.pi - math.acos(cos_angle_xoz)
                else:
                    angle_xoz = math.acos(cos_angle_xoz)
                x_zm = angle_xoz * L
                y_zm = zc

                m_zm = (h2 - (y_zm / dx))
                n_zm = (x_zm / dy)

                f_map_table[i, j] = [m_zm, n_zm]
    return f_map_table

@jit(nopython=True)
def R_maptable(h_zm, w_zm, h2, a, b, c, f, L, dx, dy, m_dot, n_dot):
    '''
    Calculate the coordinate mapping table from cylindrical unfolded image to catadioptric image.
    Args：
        h_zm: Height of cylindrical unfolded image.
        w_zm: Width of cylindrical unfolded image.
        h2: Distance from the upper edge of the cylindrical surface to the x-axis.
        a、b、c: Mirror parameters.
        f: Camera focal length.
        L: Cylindrical radius.
        dx, dy: Pixel size.
        m_dot、n_dot: Center coordinates of panoramic image, m rows and n columns.
        r_map_table：Reverse coordinate mapping table.
    '''
    r_map_table = np.zeros((h_zm, w_zm, 3), dtype=np.float64)
    for i in range(h_zm):
        for j in range(w_zm):
            x_zm = j * dx
            y_zm = (h2 - i) * dy

            angle_xoz = x_zm / L

            sin_angle_xoy = y_zm / math.sqrt(L ** 2 + y_zm ** 2)
            cos_angle_xoy = L / math.sqrt(L ** 2 + y_zm ** 2)

            tan_angle_z = (b ** 2 * cos_angle_xoy) / (2 * a * c - (a ** 2 + c ** 2) * sin_angle_xoy)

            r = f * tan_angle_z

            x_qj = r * math.cos(angle_xoz)
            y_qj = r * math.sin(angle_xoz)

            m_qj = m_dot - x_qj / dx
            n_qj = n_dot - y_qj / dy
            r = math.sqrt((x_qj / dx) ** 2 + (y_qj / dy) ** 2)

            r_map_table[i, j] = [m_qj, n_qj, r]

    return r_map_table