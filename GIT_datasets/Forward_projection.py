import numpy as np
from numba import jit
from utils import coefficient
import math

def forward(map_table,hr_img_zm,w_zm,h_zm,h_qj, w_qj,m_dot,n_dot,r1,r2):
    '''
    投影过程
    1. 获取一张全景图像
    2. 计算全景图内外圆半径，投影柱面半径L（calibration）
    3. 计算柱面展开图高度h_zm，宽度w_zm，上边缘Z轴坐标h2
    4. 获取一张彩色图像
    5. 截取高度h_zm，宽度w_zm的一张图像，作为高分辨率柱面展开图hr_img_zm
    6. 将hr_img_zm正向投影得到全景图img_qj
    :return: 全景图
    '''
    # 后向映射，用双三次插值的方法将柱面展开图像素赋值给全景图像
    @jit(nopython=True)
    def temp_Trans():
        f_tempTrans = np.zeros((h_qj, w_qj, 3), dtype=np.float64)
        for i in range(h_qj):
            for j in range(w_qj):
                # 判断像素点是否在全景图内
                if r1 <= math.sqrt(((m_dot - i) ** 2 + (n_dot - j) ** 2)) <= r2:
                    # 全景图i，j处像素坐标映射到柱面图像素坐标[m,n]
                    [m, n] = map_table[i, j]

                    # 获取n和m的整数部分和小数部分
                    m_int = int(m)  # 向下取整,行数
                    n_int = int(n)  # 向下取整,列数
                    n_frac = n - n_int
                    m_frac = m - m_int

                    # 计算行m方向和列n方向的插值系数
                    m_coeff = np.array([coefficient(m_frac + i) for i in [1, 0, -1, -2]])
                    n_coeff = np.array([coefficient(n_frac + i) for i in [1, 0, -1, -2]])

                    # 将柱面展开图坐标[m,n]周围16个像素点插值给对应全景图i，j处
                    for u in range(-1, 3):
                        for v in range(-1, 3):
                            if 0 <= (n_int + v) <= (w_zm - 1) and 0 <= (m_int + u) <= (h_zm - 1):
                                k=m_coeff[1 + u] * n_coeff[1 + v]
                                p=hr_img_zm[m_int + u, n_int + v]
                                f_tempTrans[i][j]=f_tempTrans[i][j]+ p*k

        return f_tempTrans

    tempTrans=temp_Trans()
    img_tyqj = np.array(tempTrans,dtype = 'uint8')

    return img_tyqj

