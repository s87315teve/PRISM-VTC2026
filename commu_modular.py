import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
       raise e
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera, LambertianPattern, DirectivePattern, BackscatteringPattern
from sionna.rt.utils import r_hat
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel
from sionna.phy import Block

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
# %matplotlib inline
# also try %matplotlib widget

import matplotlib.pyplot as plt

# for performance measurements
import time

# 建立通訊系統模組
class LDPC_system(Block):
    def __init__(self, num_bits_per_symbol, coderate, k, n, rg, stream_management):
        super().__init__()
        # 主要通訊參數
        self.num_bits_per_symbol=num_bits_per_symbol
        self.coderate=coderate
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.k=k
        self.n=n
        self.encoder = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(self.k, self.n)
        self.decoder = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(encoder=self.encoder, num_iter=20, return_infobits=True)
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        # OFDM相關
        self.rg=rg
        self.rg_mapper = sionna.phy.ofdm.ResourceGridMapper(self.rg)
        self.stream_management=stream_management
        self.channel=ApplyOFDMChannel(add_awgn=True)

        # channel相關
        # self.h_freq=h_freq
        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sionna.phy.ofdm.LSChannelEstimator(self.rg, interpolation_type="nn")
        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = sionna.phy.ofdm.LMMSEEqualizer(self.rg, self.stream_management)
    @tf.function() # enables graph-mode of the following function
    def call(self, batch_size, ebno_db, h_freq):
        no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        bits = self.binary_source([batch_size, 1, self.rg.num_streams_per_tx, self.k])
        # print(f"bits.shape:{bits.shape}")
        codewords = self.encoder(bits)
        # print(f"codewords.shape:{codewords.shape}")
        x = self.mapper(codewords)
        # print(f"x.shape:{x.shape}")
        x_rg = self.rg_mapper(x)
        # print(f"x_rg.shape:{x_rg.shape}")
        # y = self.awgn_channel(x_rg, no)
        y = self.channel(x_rg, h_freq, no)
        # print(f"y.shape:{y.shape}")
        h_hat, err_var = self.ls_est (y, no)
        # print(f"h_hat.shape:{h_hat.shape}")
        # print(f"err_var.shape:{err_var.shape}")
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
        # print(f"x_hat.shape:{x_hat.shape}")
        # print(f"no_eff.shape:{no_eff.shape}")
        llr = self.demapper(x_hat, no)
        # print(f"llr.shape:{llr.shape}")
        bits_hat= self.decoder(llr)
        # print(f"bits_hat.shape:{bits_hat.shape}")
        return bits, bits_hat