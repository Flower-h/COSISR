import cv2
import json
import os
import numpy as np
from utils import zm_shape
from utils import F_maptable
from utils import R_maptable
from Forward_projection import forward
from Reverse_projection import reverse
from tqdm import tqdm
from multiprocessing import Pool

'''
Forward and backward projection between catadioptric image and cylindrical unfolded image.
Args：
    COCP_path：Path to COCP-SR dataset folder.
    SIM_path： Path to SIM-SR dataset folder.
    a、b、c: Mirror parameters.
    F_L: Cylindrical radius during forward projection.
    R_L_2: Cylindrical radius during reverse projection of 2x down sampled catadioptric image.
    R_L_4: Cylindrical radius during reverse projection of 4x down sampled catadioptric image.
    f: Camera focal length.
    dx, dy: Pixel size.
    m_dot、n_dot: Center coordinates of panoramic image, m rows and n columns.
    r1: Catadioptric image inner circle radius.
    r2: Catadioptric image outer circle radius.
    h_qj: Height of catadioptric image.
    w_qj: Width of catadioptric image.
    map_table_F: Coordinate mapping table from cylindrical unfolded image projection to catadioptric image.
    map_table_R: Coordinate mapping table from catadioptric image to cylindrical unfolded image projection.
    map_table_R_2: Coordinate mapping table from 2x down sampled image to cylindrical unfolded image projection.
    map_table_R_4: Coordinate mapping table from 4x down sampled image to cylindrical unfolded image projection.
'''

COCP_path = 'COSISR/datasets/COCP-SR/'
SIM_path = 'COSISR/datasets/SIM-SR/'

a = 0.4308
b = 0.2303
c = 0.4885
dx = 0.00427
dy = 0.00427
F_L = 1.1529
R_L_2 = 0.57645
R_L_4 = 0.288147
f = [5.,2.7,1.35]
m_dot = [278,139,69]
n_dot = [277,138,69]
r1 = [55,28,14]
r2 = [270,135,68]
h_qj=[555,278,139]
w_qj=[555,278,139]

f_h_zm, f_w_zm, f_h2 = zm_shape(a, c, f[0], F_L, dx, dy, r1[0], r2[0])
r_h_zm_2, r_w_zm_2, r_h2_2 = zm_shape(a, c, f[1], R_L_2, dx, dy, r1[1], r2[1])
r_h_zm_4, r_w_zm_4, r_h2_4 = zm_shape(a, c, f[2], R_L_4, dx, dy, r1[2], r2[2])

map_table_F = F_maptable(h_qj[0], w_qj[0], f_h2, a, b, c, f[0], F_L, dx, dy, m_dot[0], n_dot[0], r1[0], r2[0])
map_table_R = R_maptable(f_h_zm, f_w_zm, f_h2, a, b, c, f[0], F_L, dx, dy, m_dot[0], n_dot[0])
map_table_R_2 = R_maptable(r_h_zm_2, r_w_zm_2, r_h2_2, a, b, c, f[1], R_L_2, dx, dy, m_dot[1], n_dot[1])
map_table_R_4 = R_maptable(r_h_zm_4, r_w_zm_4, r_h2_4, a, b, c, f[2], R_L_4, dx, dy, m_dot[2], n_dot[2])

def odi_image_projection(img_pth,output_pth, img_idx,):
    '''
    Generate cylindrical unfolded images using reverse projection of catadioptric images,
    which include HR images, 2x down sampled images, and 4x down sampled images.
    Args：
        img_pth: Path to input image.
        output_pth: Path to output image.
        img_idx: Serial number to input image.
    '''
    img_qj = cv2.imread(img_pth, flags=1)
    # 全景图下采样
    ty_img_qj_X2 = cv2.resize(img_qj, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)
    ty_img_qj_X4 = cv2.resize(img_qj, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)

    # 逆向投影(全景->lr柱面)
    hr_img_zm = reverse(map_table_R, img_qj, f_w_zm, f_h_zm, h_qj[0], w_qj[0], r1[0], r2[0])
    lr_img_zm_2 = reverse(map_table_R_2, ty_img_qj_X2, r_w_zm_2, r_h_zm_2, h_qj[1], w_qj[1], r1[1], r2[1])
    lr_img_zm_4 = reverse(map_table_R_4, ty_img_qj_X4, r_w_zm_4, r_h_zm_4, h_qj[2], w_qj[2], r1[2], r2[2])

    # 生成保存路径
    idx = img_idx
    filename = f'{idx: 04d}.png'
    root_fill = output_pth
    HR_path = root_fill + '/HR'
    X2_path = root_fill + '/LR/X2'
    X4_path = root_fill + '/LR/X4'
    os.makedirs(HR_path, exist_ok=True)
    os.makedirs(X2_path, exist_ok=True)
    os.makedirs(X4_path, exist_ok=True)

    cv2.imwrite(HR_path + "/" + filename, hr_img_zm)
    cv2.imwrite(X2_path + "/" + filename, lr_img_zm_2)
    cv2.imwrite(X4_path + "/" + filename, lr_img_zm_4)



# 开始生成新数据集
def extract_odi_image(img_pth, output_pth, img_idx, split_option, ):
    '''
    Generate 2x and 4x downsampled cylindrical unfolded images by forward and backward
    projection of pseudo cylindrical unfolded images.
    Args：
        img_pth: Path to input image.
        output_pth: Path to output image.
        img_idx: Serial number to input image.
        split_option: train/test/validation.
    '''
    # 导入原始图片
    img = cv2.imread(img_pth, flags=1)
    # 获取图像尺寸
    img_height, img_width, channels = img.shape

    # 判断原始图片尺寸是否符合要求1696*480
    if img_width > 1696 and img_height > 480:
        img_zm_list = []
        idx = 0

        # 生成HR-LR图像，以及投影得到的全景图
        # 裁剪出尺寸为[h_zm,w_zm]的HR柱面展开图
        left = img_width - f_w_zm - 1
        top = img_height - f_h_zm - 1
        # 裁剪不同位置
        if split_option == 'train' or split_option == 'test':
            left = int(left/3)
            top = int(top/3)
            right = left + f_w_zm
            bottom = top + f_h_zm
            hr_img1_zm = img[top:bottom, left:right]
            hr_img1_zm = np.ascontiguousarray(hr_img1_zm)
            img_zm_list.append(hr_img1_zm)

        if split_option == 'train' or split_option == 'validation':
            left = int(left*2/3)
            top = int(top*2/3)
            right = left + f_w_zm
            bottom = top + f_h_zm
            hr_img2_zm = img[top:bottom, left:right]
            hr_img2_zm = np.ascontiguousarray(hr_img2_zm)
            img_zm_list.append(hr_img2_zm)

        if split_option == 'train':
            idx = img_idx * 2
        elif split_option == 'validation':
            idx = img_idx
        elif split_option == 'test':
            idx = img_idx


        for hr_img_zm in img_zm_list:
            # 正向投影(hr柱面->全景)
            ty_img_qj = forward(map_table_F, hr_img_zm, f_w_zm, f_h_zm, h_qj[0], w_qj[0], m_dot[0], n_dot[0], r1[0],
                                 r2[0])

            # 全景图下采样
            ty_img_qj_X2 = cv2.resize(ty_img_qj, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)  # X2
            ty_img_qj_X4 = cv2.resize(ty_img_qj, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)  # X4

            # 逆向投影(全景->lr柱面)
            lr_img_zm = reverse(map_table_R, ty_img_qj, f_w_zm, f_h_zm, h_qj[0], w_qj[0], r1[0], r2[0])
            lr_img_zm_2 = reverse(map_table_R_2, ty_img_qj_X2, r_w_zm_2, r_h_zm_2, h_qj[1], w_qj[1], r1[1], r2[1])
            lr_img_zm_4 = reverse(map_table_R_4, ty_img_qj_X4, r_w_zm_4, r_h_zm_4, h_qj[2], w_qj[2], r1[2], r2[2])

            # 保存图像
            filename = f'{idx: 04d}.png'
            root_fill = output_pth
            HR_path = root_fill + '/HR'
            C_path = root_fill + '/Catadioptric'
            X2_path = root_fill + '/LR/X2'
            X4_path = root_fill + '/LR/X4'
            os.makedirs(HR_path, exist_ok=True)
            os.makedirs(C_path, exist_ok=True)
            os.makedirs(X2_path, exist_ok=True)
            os.makedirs(X4_path, exist_ok=True)

            cv2.imwrite(HR_path + "/" + filename, hr_img_zm)
            cv2.imwrite(C_path + "/" + filename, ty_img_qj)
            cv2.imwrite(X2_path + "/" + filename, lr_img_zm_2)
            cv2.imwrite(X4_path + "/" + filename, lr_img_zm_4)

            idx += 1

def main():
    '''
    Generate COCP-SR dataset and SIM-SR dataset
    '''

    #COCP-SR
    print("Generating COCP-SR dataset.\n")

    #train
    with open(os.path.join(COCP_path, 'train_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating train dataset with %d images in total.\n" % len(img_list))

    output_train = COCP_path + 'train'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(odi_image_projection, args=(path,output_train, i,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("Train dataset generation completed.\n")

    #test
    with open(os.path.join(COCP_path, 'test_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating test dataset with %d images in total.\n" % len(img_list))

    output_test = COCP_path + 'test'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(odi_image_projection, args=(path, output_test, i,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("Test dataset generation completed.\n")

    #validation
    with open(os.path.join(COCP_path, 'val_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating validation dataset with %d images in total.\n" % len(img_list))

    output_val = COCP_path + 'validation'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(odi_image_projection, args=(path, output_val, i,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("validation dataset generation completed.\n")
    print("All content of COCP-SR dataset has been generated.\n")


    #SIM-SR
    print("Generating SIM-SR dataset.\n")

    #train
    with open(os.path.join(SIM_path, 'train_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating train dataset with %d images in total.\n" % len(img_list))

    split_option = 'train'
    output_train = SIM_path + 'train'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(extract_odi_image, args=(path, output_train, i, split_option,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("Train dataset generation completed.\n")

    #test
    with open(os.path.join(SIM_path, 'val_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating test dataset with %d images in total.\n" % len(img_list))

    split_option = 'test'
    output_train = SIM_path + 'test'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(extract_odi_image, args=(path, output_train, i, split_option,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("test dataset generation completed.\n")

    #val
    with open(os.path.join(SIM_path, 'val_images.json'), 'r') as j:
        img_list = json.load(j)
    print("Generating validation dataset with %d images in total.\n" % len(img_list))

    split_option = 'validation'
    output_train = SIM_path + 'validation'
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(20)
    for i, path in enumerate(img_list):
        pool.apply_async(extract_odi_image, args=(path, output_train, i, split_option,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print("validation dataset generation completed.\n")
    print("All content of SIM-SR dataset has been generated.\n")

if __name__ == '__main__':
    main()