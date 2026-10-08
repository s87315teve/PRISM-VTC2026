import pandas as pd
import json
import numpy as np
import mitsuba as mi
import drjit as dr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime

# # 設定 Mitsuba3 變體
# mi.set_variant("cuda_ad_mono_polarized")

import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, \
                      PathSolver, r_hat, PolarizedAntennaPattern, \
                      register_antenna_pattern, AntennaPattern

class Trainable_Fourier_V_Pattern:
    """
    傅立葉級數展開版本：更有彈性，能擬合各種場型
    
    場型：Σ a_k * cos(k*θ)
    """
    def __init__(self, opt: mi.ad.Optimizer, n_terms=4):
        self.n_terms = n_terms
        
        # 初始化傅立葉係數
        for k in range(n_terms):
            opt[f"a_{k}"] = mi.Float(0.1 if k > 0 else 0.5)
        
        self.opt = opt
    
    def __call__(self, theta, phi):
        amplitude = mi.Float(0.0)
        
        for k in range(self.n_terms):
            amplitude = amplitude + self.opt[f"a_{k}"] * dr.cos(k * theta)
        
        # 確保非負
        amplitude = dr.abs(amplitude)
        
        return mi.Complex2f(amplitude, 0)

def trainable_fourier_factory(*, opt, n_terms=4, polarization="V", polarization_model="tr38901_2"):
    """傅立葉版：最有彈性"""
    return PolarizedAntennaPattern(
        v_pattern=Trainable_Fourier_V_Pattern(opt, n_terms=n_terms),
        polarization=polarization,
        polarization_model=polarization_model
    )

register_antenna_pattern("trainable_fourier", trainable_fourier_factory)

# load example
# scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
#                              pattern="trainable_fourier",
#                              opt=opt,
#                              polarization="V")



def elliptical_beam_pattern_func(theta: mi.Float, phi: mi.Float, 
                                  theta_0: float, phi_0: float,
                                  theta_width: float, phi_width: float) -> mi.Complex2f:
    """
    橢圓波束場型函數
    
    參數:
        theta: 天頂角
        phi: 方位角
        theta_0: 主波束天頂角方向 (弧度)
        phi_0: 主波束方位角方向 (弧度)
        theta_width: theta 方向的波束寬度參數 (弧度)
        phi_width: phi 方向的波束寬度參數 (弧度)
    """
    # 計算相對於主波束方向的角度偏移
    delta_theta = theta - theta_0
    delta_phi = phi - phi_0
    
    # 處理 phi 的週期性 (-π 到 π)
    delta_phi = dr.atan2(dr.sin(delta_phi), dr.cos(delta_phi))
    
    # 使用高斯函數，只在主波束方向有峰值
    amplitude = dr.exp(-0.5 * ((delta_theta / theta_width)**2 + 
                                (delta_phi / phi_width)**2))
    
    return mi.Complex2f(amplitude, 0)

class EllipticalBeamPattern(AntennaPattern):
    def __init__(self, theta_0=90.0, phi_0=0.0, 
                 theta_beamwidth=30.0, phi_beamwidth=60.0):
        """
        參數:
            theta_0: 主波束天頂角方向 (度)
            phi_0: 主波束方位角方向 (度)
            theta_beamwidth: theta 方向半功率波束寬度 (度)
            phi_beamwidth: phi 方向半功率波束寬度 (度)
        """
        theta_0_rad = np.deg2rad(theta_0)
        phi_0_rad = np.deg2rad(phi_0)
        # 將波束寬度轉換為高斯函數的標準差
        theta_width = np.deg2rad(theta_beamwidth) / 2.355
        phi_width = np.deg2rad(phi_beamwidth) / 2.355
        
        def my_pattern(theta, phi):
            """返回 (c_theta, c_phi)"""
            c_theta = elliptical_beam_pattern_func(theta, phi, 
                                                   theta_0_rad, phi_0_rad,
                                                   theta_width, phi_width)
            c_phi = dr.zeros(mi.Complex2f, dr.width(c_theta))
            return c_theta, c_phi
        
        # 建立 patterns 屬性
        self.patterns = [my_pattern]

def elliptical_beam_factory(theta_0=90.0, phi_0=0.0, 
                            theta_beamwidth=30.0, phi_beamwidth=60.0):
    """工廠函數"""
    return EllipticalBeamPattern(theta_0, phi_0, 
                                 theta_beamwidth, phi_beamwidth)

# 註冊天線場型
register_antenna_pattern("elliptical_beam", elliptical_beam_factory)

# load example
# array = PlanarArray(num_rows=1, num_cols=1, 
#                    pattern="elliptical_beam",
#                    theta_0=90,           # 主波束水平
#                    phi_0=30.0,              # 方位角 30°
#                    theta_beamwidth=30.0,   # theta 方向窄
#                    phi_beamwidth=30.0)     # phi 方向寬
