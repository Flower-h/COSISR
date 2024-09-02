from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .utils import paired_random_crop
import numpy as np
import cv2
import torch
import os.path as osp


@DATASET_REGISTRY.register()
class COIPairedImageDataset(data.Dataset):
    """Paired image dataset for conditional ODI restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(COIPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        #获取所有图片的路径列表paths
        if 'ext_dataroot_gt' in self.opt:
            assert self.io_backend_opt['type'] == 'disk'
            self.ext_gt_folder, self.ext_lq_folder = opt['ext_dataroot_gt'], opt['ext_dataroot_lq']
            if 'enlarge_scale' in self.opt:
                enlarge_scale = self.opt['enlarge_scale']
            else:
                enlarge_scale = [1 for _ in range(len(self.ext_gt_folder)+1)]

            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl) \
                         * enlarge_scale[0]
            for i in range(len(self.ext_gt_folder)):
                self.paths += paired_paths_from_folder([self.ext_lq_folder[i], self.ext_gt_folder[i]], ['lq', 'gt'],
                                                          self.filename_tmpl) * enlarge_scale[i+1]
        #无'ext_dataroot_gt' 的code
        else:
            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                              self.opt['meta_info_file'], self.filename_tmpl)
            else:
                self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        #这里设置畸变拉伸Cd：osrt_Cd=cos((2m+1-M)/M * pi/2)
        #gt_h: 1024  gt_w: 2048,未分割的完整HR图片尺寸，建立整幅图的畸变条件：condition
        if 'gt_size' in self.opt and self.opt['gt_size']:
            self.glob_condition = get_condition(self.opt['gt_h']//self.opt['scale'], self.opt['gt_w']//self.opt['scale'], self.opt['condition_type'])

        if 'sub_image' in self.opt and self.opt['sub_image']:
            self.sub_image = True
        else:
            self.sub_image = False


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        #获取HR图片在原图上左上角坐标sub_h, sub_w，并计算对应LR图片在原图上左上角坐标sub_h, sub_w
        if self.sub_image:
            sub_h, sub_w = osp.split(lq_path)[-1].split('_')[3:5]
            sub_h, sub_w = int(sub_h) // scale, int(sub_w) // scale
        else:
            sub_h, sub_w = 0, 0

        if self.opt.get('force_resize'):
        # resize gt with wrong resolutions
            img_gt = cv2.resize(img_gt, (img_lq.shape[1] * scale, img_lq.shape[0] * scale), cv2.INTER_CUBIC)

        # augmentation for training
        # random crop
        if 'gt_size' in self.opt and self.opt['gt_size']:
            gt_size = self.opt['gt_size']
            #从输入的img_gt中截取出gt_size大小的图片作为训练的HR patch, img_lq中截取出gt_size/scale大小的图片作为训练的LR patch，
            # 并得到HR patch和LR patch在整个源图上的左上角坐标
            img_gt, img_lq, top_lq, left_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path,
                                                                       return_top_left=True)
            # top = int(top_lq * scale)
            # top = top + int(sub_h * scale)
            top_lq, left_lq = top_lq + sub_h, left_lq + sub_w
            if self.opt['condition_type'] is not None:
                if ('DIV2K' or 'Flickr2K') in lq_path:
                    _condition = torch.zeros([1, img_lq.shape[0], img_lq.shape[1]])
                else:
                    #从整个畸变图上截取出与img_lq相同位置的畸变图
                    _condition = self.glob_condition[:,top_lq:top_lq+img_lq.shape[0],left_lq:left_lq+img_lq.shape[1]]
            else:
                _condition = 0.
        else:
            _condition = get_condition(img_lq.shape[0], img_lq.shape[1], self.opt['condition_type'])
        #使用翻转、旋转增强数据集
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 源code
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'condition': _condition}

    def __len__(self):
        return len(self.paths)

def get_condition(h, w, condition_type):
    if condition_type is None:
        return 0.
    elif condition_type == 'X1':
        m0 = 170
        L = 1.1529
        f = 5.
        return make_coord((h, w), m0, L, f).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1)
    elif condition_type == 'X2':
        m0 = 73
        L = 0.57645
        f = 2.7
        return make_coord((h, w), m0, L, f).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1)
    elif condition_type == 'X4':
        m0 = 37
        L = 0.288147
        f = 1.35
        return make_coord((h, w), m0, L, f).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1)
    else:
        raise RuntimeError('Unsupported condition type')


def make_coord(shape, m0, L, f, flatten=False):
    """ Make coordinates at grid centers.
    """
    a = 0.4308
    b = 0.2303
    c = 0.4885
    dx = 0.00427
    h = shape[0]
    L = L * torch.ones(h)
    H = a ** 2 + c ** 2
    J = 2 * a * c
    y = (m0 * torch.ones(h) - torch.arange(h).float()) * dx
    P = torch.sqrt(L ** 2 + y ** 2)
    seq_x = f * b ** 2 / (J * P - H * y)
    seq_y = f * b ** 2 * L * (H*P-J*y)/((J * P - H*y)**2 * P)
    ret = torch.stack((seq_x,seq_y), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret




#
# def get_condition(h, w, condition_type):
#     if condition_type is None:
#         return 0.
#     elif condition_type == 'cos_latitude':
#         return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)
#     elif condition_type == 'latitude':
#         return make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2, 0, 1) * math.pi / 2
#     elif condition_type == 'coord':
#         return make_coord([h, w]).permute(2, 0, 1)
#     else:
#         raise RuntimeError('Unsupported condition type')
#
#
# def make_coord(shape, ranges=(-1, 1), flatten=False):
#     """ Make coordinates at grid centers.
#     """
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         v0, v1 = ranges
#         r = (v1 - v0) / (2 * n)
#         seq = v0 + r + (2 * r) * torch.arange(n).float()
#         coord_seqs.append(seq)
#     ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
#     if flatten:
#         ret = ret.view(-1, ret.shape[-1])
#     return ret

