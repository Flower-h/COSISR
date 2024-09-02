import numpy as np
from numba import jit
from utils import coefficient

def reverse(map_table,img_qj,w_zm,h_zm,h_qj, w_qj,r1,r2):
    # 前向映射，用双三次插值的方法将全景图像像素赋值给柱面展开图
    @jit(nopython=True)
    def temp_Trans(h_zm, w_zm):
        # 创建一个空白的图像，用于存储投影得到的柱面展开图像
        tempTrans = np.zeros((h_zm, w_zm, 3), dtype=np.float64)
        # 前向映射，用双三次插值的方法将全景图像像素赋值给柱面展开图
        for i in range(h_zm):
            for j in range(w_zm):
                [m, n, r] = map_table[i, j]
                # 判断像素点是否在全景图内
                if r1 <= r <= r2:
                    # 全景图i，j处像素坐标映射到柱面图像素坐标[m,n]
                    # 获取n和m的整数部分和小数部分
                    m_int = int(m)  # 向下取整,行数
                    n_int = int(n)  # 向下取整,列数
                    m_frac = m - m_int
                    n_frac = n - n_int

                    # 双线性插值
                    k = [m_frac, 1 - m_frac, n_frac, 1 - n_frac]
                    tempTrans[i, j] = img_qj[m_int, n_int] * k[0] * k[2] + \
                                      img_qj[m_int, n_int + 1] * k[0] * k[3] + \
                                      img_qj[m_int + 1, n_int] * k[1] * k[2] + \
                                      img_qj[m_int + 1, n_int + 1] * k[1] * k[3]

                    # #双三次插值
                    # # 计算行m方向和列n方向的插值系数
                    # m_coeff = np.array([coefficient(m_frac + i) for i in [1, 0, -1, -2]])
                    # n_coeff = np.array([coefficient(n_frac + i) for i in [1, 0, -1, -2]])
                    # # 将全景图坐标[m,n]周围16个像素点插值给对应柱面图i，j处
                    # for u in range(-1, 3):
                    #     for v in range(-1, 3):
                    #         if 0 <= (n_int + v) <= (w_qj - 1) and 0 <= (m_int + u) <= (h_qj - 1):
                    #             k = m_coeff[1 + u] * n_coeff[1 + v]
                    #             p = img_qj[m_int + u, n_int + v]
                    #             tempTrans[i,j] = tempTrans[i,j] + p * k


        return tempTrans

    temp_zm = temp_Trans(h_zm, w_zm)
    img_zm = np.array(temp_zm, dtype='uint8')
    return img_zm