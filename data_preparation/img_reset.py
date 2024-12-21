import math
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool

input_path = 'COSISR/datasets/data'
def main():
    '''
    input_path: Path to the input image
    train_path: Path to the training set images
    test_path: Path to the test set image
    val_path: Path to the validation set image
    '''
    train_path = 'COSISR/datasets/COCP-SR/train/Catadioptric'
    test_path = 'COSISR/datasets/COCP-SR/test/Catadioptric'
    val_path = 'COSISR/datasets/COCP-SR/validation/Catadioptric'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    img_list = os.listdir(input_path)
    img_list.sort(key=lambda x:int(x.split('.')[0]))

    #train
    lenx = len(img_list)
    d0 = lenx - lenx % 10
    d = lenx % 10
    if d < 5:
        train_len = d0 / 2 + d
    else:
        train_len = d0 / 2 + d - d % 5
    print("Reserve %d images as COCP-SR training dataset after preprocessing of public dataset\n" % train_len)

    pbar = tqdm(total=train_len, unit='image', desc='Extract')
    pool = Pool(20)
    idx = 0
    for i, path in enumerate(img_list):
        pool.apply_async(res_catadioptric, args=(path, idx, train_path,), callback=lambda arg: pbar.update(1))
        idx += 1
        if idx == train_len:
            break
    pool.close()
    pool.join()
    pbar.close()
    print("There are %d images in the train dataset\n" % idx)

    #val
    val_len = 100
    pbar = tqdm(total=val_len, unit='image', desc='Extract')
    pool = Pool(20)
    idx = 0
    for i, path in enumerate(img_list):
        if i > train_len:
            if i % 10 == 6:
                pool.apply_async(res_catadioptric, args=(path, idx, val_path,), callback=lambda arg: pbar.update(1))
                idx += 1
            if idx == 100:
                break
    pool.close()
    pool.join()
    pbar.close()
    print("There are %d images in the validation dataset\n" % idx)

    #test
    test_len = 100
    pbar = tqdm(total=test_len, unit='image', desc='Extract')
    pool = Pool(20)
    idx = 0
    for i, path in enumerate(img_list):
        if i % 10 == 6:
            pool.apply_async(res_catadioptric, args=(path, idx, test_path,), callback=lambda arg: pbar.update(1))
            idx += 1
        if idx == 100:
            break
    pool.close()
    pool.join()
    pbar.close()
    print("There are %d images in the test dataset\n" % idx)

def res_catadioptric(path, idx, output_path, ):
    '''
    Preprocessing of Original Images:
        Determine the cutting position imgcrop
        Crop the image into a square shape
        Scale the size to 555 * 555
    '''
    filename = f'{idx: 04d}.png'
    img_path = os.path.join(input_path, path)

    img = cv2.imread(img_path)
    imgcrop = img[26:1396, 12:1382]

    imgres = cv2.resize(imgcrop, (555, 555))
    cv2.imwrite(output_path + "/" + filename, imgres)

if __name__ == '__main__':
    main()