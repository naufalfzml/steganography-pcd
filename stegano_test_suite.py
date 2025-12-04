import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import string
import os

# Import Real Classes
from stegano_edge import SteganographyEdge
from stegano_edge_clustered import SteganographyEdgeClustered
from stegano_edge_adaptive import SteganographyEdgeAdaptive

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
TEST_IMAGE_PATH = "images/bridge.tiff"
EDGE_THRESHOLD = 60
ADP_EPS = 0.2
ADP_MIN_SAMPLES = 3
ADP_USE_ISOLATED = False
ADP_VAR_PERCENTILE = 90
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# BAGIAN 1: KELAS ALGORITMA (WRAPPER)
# =============================================================================

class StandardEdgeStego:
    def get_capacity(self, image):
        cv2.imwrite("temp_cap_std.png", image)
        success, res = SteganographyEdge.get_capacity("temp_cap_std.png", threshold=EDGE_THRESHOLD)
        if success:
            return res['max_chars']
        return 0

    def embed(self, image, message):
        cv2.imwrite("temp_src_std.png", image)
        output_path = "temp_out_std.png"
        
        if os.path.exists(output_path): 
            os.remove(output_path)
            
        success, msg = SteganographyEdge.encode_message(
            "temp_src_std.png", message, output_path, threshold=EDGE_THRESHOLD
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image

    def extract(self, stego_image, expected_length=None):
        cv2.imwrite("temp_ext_std.png", stego_image)
        success, msg = SteganographyEdge.decode_message(
            "temp_ext_std.png", 
            threshold=EDGE_THRESHOLD,
            expected_length=expected_length
        )
        if success:
            return msg
        return ""

class ClusteredEdgeStego:
    def __init__(self):
        self.eps = ADP_EPS
        self.min_samples = ADP_MIN_SAMPLES
        self.threshold = EDGE_THRESHOLD
        self.use_isolated = ADP_USE_ISOLATED 

    def get_capacity(self, image):
        cv2.imwrite("temp_cap_clu.png", image)
        success, res = SteganographyEdgeClustered.get_capacity(
            "temp_cap_clu.png", self.threshold, self.eps, self.min_samples, 
            self.use_isolated
        )
        if success:
            return res['max_chars']
        return 0

    def embed(self, image, message):
        cv2.imwrite("temp_src_clu.png", image)
        output_path = "temp_out_clu.png"
        
        if os.path.exists(output_path): 
            os.remove(output_path)
        
        success, msg = SteganographyEdgeClustered.encode_message(
            "temp_src_clu.png", message, output_path, 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image

    def extract(self, stego_image, expected_length=None):
        cv2.imwrite("temp_ext_clu.png", stego_image)
        success, msg = SteganographyEdgeClustered.decode_message(
            "temp_ext_clu.png", 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated,
            expected_length=expected_length
        )
        if success:
            return msg
        return ""

class AdaptiveEdgeStego:
    def __init__(self):
        self.eps = ADP_EPS
        self.min_samples = ADP_MIN_SAMPLES
        self.threshold = EDGE_THRESHOLD
        self.use_isolated = ADP_USE_ISOLATED 
        self.variance_percentile = ADP_VAR_PERCENTILE

    def get_capacity(self, image):
        cv2.imwrite("temp_cap_adp.png", image)
        success, res = SteganographyEdgeAdaptive.get_capacity(
            "temp_cap_adp.png", self.threshold, self.eps, self.min_samples, 
            self.use_isolated, self.variance_percentile
        )
        if success:
            return res['max_chars']
        return 0

    def embed(self, image, message):
        cv2.imwrite("temp_src_adp.png", image)
        output_path = "temp_out_adp.png"
        
        if os.path.exists(output_path): 
            os.remove(output_path)
        
        success, msg = SteganographyEdgeAdaptive.encode_message(
            "temp_src_adp.png", message, output_path, 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated, self.variance_percentile
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image

    def extract(self, stego_image, expected_length=None):
        cv2.imwrite("temp_ext_adp.png", stego_image)
        success, msg = SteganographyEdgeAdaptive.decode_message(
            "temp_ext_adp.png", 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated,
            expected_length=expected_length
        )
        if success:
            return msg
        return ""

# =============================================================================
# BAGIAN 2: UTILITIES & METRICS
# =============================================================================

class Utils:
    @staticmethod
    def generate_random_string(length):
        return 'a' * length

    @staticmethod
    def str_to_bin(message):
        return "".join(f"{ord(c):08b}" for c in message)

    @staticmethod
    def calculate_ber(original_msg, extracted_msg):
        bin_orig = Utils.str_to_bin(original_msg)
        bin_extr = Utils.str_to_bin(extracted_msg)
        
        len_orig = len(bin_orig)
        len_extr = len(bin_extr)
        
        if len_orig == 0:
            return 0.0
        
        error_count = 0
        
        for i in range(len_orig):
            if i >= len_extr:
                error_count += 1
            elif bin_orig[i] != bin_extr[i]:
                error_count += 1
        
        ber = (error_count / len_orig) * 100.0
        
        return min(ber, 100.0)

    @staticmethod
    def add_salt_pepper_noise(image, prob):
        output = np.copy(image)
        prob_pepper = prob / 2
        prob_salt = prob / 2
        thres_salt = 1 - prob_salt
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                
                if rdn < prob_pepper:
                    output[i][j] = [0, 0, 0]
                elif rdn > thres_salt:
                    output[i][j] = [255, 255, 255]
        
        return output

    @staticmethod
    def apply_jpeg_compression(image, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        output = np.copy(image).astype(np.float64)
        gaussian = np.random.normal(mean, sigma, image.shape)
        output = output + gaussian
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    @staticmethod
    def add_rayleigh_noise(image, scale=25):
        output = np.copy(image).astype(np.float64)
        rayleigh = np.random.rayleigh(scale, image.shape)
        rayleigh = rayleigh - np.mean(rayleigh)
        output = output + rayleigh
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    @staticmethod
    def add_erlang_noise(image, shape=2, scale=15):
        output = np.copy(image).astype(np.float64)
        erlang = np.random.gamma(shape, scale, image.shape)
        erlang = erlang - np.mean(erlang)
        output = output + erlang
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    @staticmethod
    def add_uniform_noise(image, low=-50, high=50):
        output = np.copy(image).astype(np.float64)
        uniform = np.random.uniform(low, high, image.shape)
        output = output + uniform
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    @staticmethod
    def add_exponential_noise(image, scale=30):
        output = np.copy(image).astype(np.float64)
        exponential = np.random.exponential(scale, image.shape)
        exponential = exponential - np.mean(exponential)
        output = output + exponential
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

# =============================================================================
# BAGIAN 3: AUTOMATED TESTING FRAMEWORK
# =============================================================================

class StegoAutomatedTester:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print("⚠ Gambar tidak ditemukan. Membuat dummy image gradient...")
            self.original_image = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(512):
                self.original_image[i, :] = (i % 256, (i*2) % 256, (255-i) % 256)
        
        self.algo_standard = StandardEdgeStego()
        self.algo_clustered = ClusteredEdgeStego()
        self.algo_adaptive = AdaptiveEdgeStego()
        
        self.results_payload = []
        self.results_robustness = []
        
        # ✅ HITUNG KAPASITAS DALAM CHARS DAN BITS
        self.capacity_std_chars = self.algo_standard.get_capacity(self.original_image)
        self.capacity_clu_chars = self.algo_clustered.get_capacity(self.original_image)
        self.capacity_adp_chars = self.algo_adaptive.get_capacity(self.original_image)
        
        # Convert ke bits (termasuk overhead header/delimiter)
        self.capacity_std_bits = self.capacity_std_chars * 8 + 56  # +56 bits delimiter "<<END>>"
        self.capacity_clu_bits = self.capacity_clu_chars * 8 + 32  # +32 bits header length
        self.capacity_adp_bits = self.capacity_adp_chars * 8 + 32  # +32 bits header length
        
        # ✅ HITUNG BPP (Bits Per Pixel)
        h, w = self.original_image.shape[:2]
        self.total_pixels = h * w
        
        self.bpp_std = self.capacity_std_bits / self.total_pixels
        self.bpp_clu = self.capacity_clu_bits / self.total_pixels
        self.bpp_adp = self.capacity_adp_bits / self.total_pixels
        
        # Display capacity info dengan BITS sebagai primary
        print(f"\n📊 Kapasitas Maksimal Gambar Original ({w}×{h} = {self.total_pixels:,} pixels)")
        print("="*95)
        print(f"{'Method':<20} | {'Capacity (bits)':<18} | {'bpp':<10} | {'Chars':<12} | {'Overhead':<10}")
        print("-"*95)
        print(f"{'Standard Edge':<20} | {self.capacity_std_bits:>17,} | {self.bpp_std:>9.4f} | "
              f"{self.capacity_std_chars:>11,} | 56 bits")
        print(f"{'Clustered Edge':<20} | {self.capacity_clu_bits:>17,} | {self.bpp_clu:>9.4f} | "
              f"{self.capacity_clu_chars:>11,} | 16 bits")
        print(f"{'Adaptive Edge':<20} | {self.capacity_adp_bits:>17,} | {self.bpp_adp:>9.4f} | "
              f"{self.capacity_adp_chars:>11,} | 32 bits")
        print("="*95)
        print("ℹ️  bpp (bits per pixel) = metrik standar untuk publikasi ilmiah")
        print("ℹ️  Overhead = bits untuk header/delimiter (bukan data payload)")

    def run_payload_stress_test(self):
        print("\n[1/2] Menjalankan Payload Capacity Stress Test...")
        print("ℹ️  Format: Utilisasi% (payload_bits/capacity_bits) | PSNR | SSIM\n")
        
        # Gunakan kapasitas standard sebagai baseline 100%
        max_cap_chars = self.capacity_std_chars
        max_cap_bits = self.capacity_std_bits
        
        payload_percentages = range(10, 101, 10)
        
        for pct in payload_percentages:
            msg_len = int(max_cap_chars * (pct / 100.0))
            payload_bits = msg_len * 8
            message = Utils.generate_random_string(msg_len)
            
            # --- Test Standard ---
            ret_s, stego_s = self.algo_standard.embed(self.original_image, message)
            if ret_s:
                p_s = psnr(self.original_image, stego_s)
                s_s = ssim(self.original_image, stego_s, channel_axis=2)
                ext_s = self.algo_standard.extract(stego_s)
                ber_s = Utils.calculate_ber(message, ext_s)
            else:
                p_s, s_s, ber_s = 0, 0, 100

            # --- Test Clustered ---
            ret_c, stego_c = self.algo_clustered.embed(self.original_image, message)
            if ret_c:
                p_c = psnr(self.original_image, stego_c)
                s_c = ssim(self.original_image, stego_c, channel_axis=2)
                ext_c = self.algo_clustered.extract(stego_c)
                ber_c = Utils.calculate_ber(message, ext_c)
            else:
                p_c, s_c, ber_c = 0, 0, 100

            # --- Test Adaptive ---
            ret_a, stego_a = self.algo_adaptive.embed(self.original_image, message)
            if ret_a:
                p_a = psnr(self.original_image, stego_a)
                s_a = ssim(self.original_image, stego_a, channel_axis=2)
                ext_a = self.algo_adaptive.extract(stego_a)
                ber_a = Utils.calculate_ber(message, ext_a)
            else:
                p_a, s_a, ber_a = 0, 0, 100

            # ✅ HITUNG UTILISASI UNTUK SETIAP ALGORITMA
            util_std = (payload_bits / self.capacity_std_bits) * 100
            util_clu = (payload_bits / self.capacity_clu_bits) * 100
            util_adp = (payload_bits / self.capacity_adp_bits) * 100

            self.results_payload.append({
                'Payload (%)': pct,
                'Payload_Bits': payload_bits,
                'Payload_Chars': msg_len,
                'Std_Util': util_std,
                'Std_PSNR': p_s, 
                'Std_SSIM': s_s,
                'Clu_Util': util_clu,
                'Clu_PSNR': p_c, 
                'Clu_SSIM': s_c,
                'Adp_Util': util_adp,
                'Adp_PSNR': p_a, 
                'Adp_SSIM': s_a
            })
            
            print(f"   {pct:>3}% | {payload_bits:>8,} bits ({msg_len:>6,} chars)")
            print(f"        Std: {util_std:>5.1f}% ({payload_bits:>8,}/{self.capacity_std_bits:>8,}) | "
                  f"PSNR: {p_s:>6.2f}dB | SSIM: {s_s:.4f}")
            print(f"        Clu: {util_clu:>5.1f}% ({payload_bits:>8,}/{self.capacity_clu_bits:>8,}) | "
                  f"PSNR: {p_c:>6.2f}dB | SSIM: {s_c:.4f}")
            print(f"        Adp: {util_adp:>5.1f}% ({payload_bits:>8,}/{self.capacity_adp_bits:>8,}) | "
                  f"PSNR: {p_a:>6.2f}dB | SSIM: {s_a:.4f}")
            print()

    def run_robustness_test(self):
        print("\n[2/2] Menjalankan Robustness Test...")
        
        # Fix payload 50% dari Standard capacity
        payload_chars = int(self.capacity_std_chars * 0.5)
        payload_bits = payload_chars * 8
        message = Utils.generate_random_string(payload_chars)
        
        print(f"   ℹ️  Payload: {payload_bits:,} bits ({payload_chars:,} chars)")
        print(f"   ℹ️  Utilisasi Kapasitas:")
        util_std = (payload_bits / self.capacity_std_bits) * 100
        util_clu = (payload_bits / self.capacity_clu_bits) * 100
        util_adp = (payload_bits / self.capacity_adp_bits) * 100
        print(f"      - Standard : {util_std:>5.1f}% ({payload_bits:>8,}/{self.capacity_std_bits:>8,} bits)")
        print(f"      - Clustered: {util_clu:>5.1f}% ({payload_bits:>8,}/{self.capacity_clu_bits:>8,} bits)")
        print(f"      - Adaptive : {util_adp:>5.1f}% ({payload_bits:>8,}/{self.capacity_adp_bits:>8,} bits)")
        
        # Generate Base Stego Images
        _, stego_std = self.algo_standard.embed(self.original_image, message)
        _, stego_clu = self.algo_clustered.embed(self.original_image, message)
        _, stego_adp = self.algo_adaptive.embed(self.original_image, message)
        
        attacks = [
            ("No Attack", lambda x: x),
            ("Gaussian (σ=15)", lambda x: Utils.add_gaussian_noise(x, mean=0, sigma=15)),
            ("Gaussian (σ=25)", lambda x: Utils.add_gaussian_noise(x, mean=0, sigma=25)),
            ("Gaussian (σ=35)", lambda x: Utils.add_gaussian_noise(x, mean=0, sigma=35)),
            ("Rayleigh (scale=15)", lambda x: Utils.add_rayleigh_noise(x, scale=15)),
            ("Rayleigh (scale=25)", lambda x: Utils.add_rayleigh_noise(x, scale=25)),
            ("Erlang (k=2,s=15)", lambda x: Utils.add_erlang_noise(x, shape=2, scale=15)),
            ("Erlang (k=3,s=15)", lambda x: Utils.add_erlang_noise(x, shape=3, scale=15)),
            ("Uniform (-30,+30)", lambda x: Utils.add_uniform_noise(x, low=-30, high=30)),
            ("Uniform (-50,+50)", lambda x: Utils.add_uniform_noise(x, low=-50, high=50)),
            ("Exponential (s=25)", lambda x: Utils.add_exponential_noise(x, scale=25)),
            ("Salt&Pepper (0.5%)", lambda x: Utils.add_salt_pepper_noise(x, 0.005)),
            ("Salt&Pepper (1.0%)", lambda x: Utils.add_salt_pepper_noise(x, 0.01)),
            ("Salt&Pepper (2.0%)", lambda x: Utils.add_salt_pepper_noise(x, 0.02)),
            ("JPEG (Q=90)", lambda x: Utils.apply_jpeg_compression(x, 90)),
            ("JPEG (Q=70)", lambda x: Utils.apply_jpeg_compression(x, 70)),
            ("JPEG (Q=50)", lambda x: Utils.apply_jpeg_compression(x, 50)),
        ]
        
        print(f"\n{'Attack Type':<30} | {'Std BER':<10} | {'Clu BER':<10} | {'Adp BER':<10} | {'Winner':<10}")
        print("-" * 85)
        
        for attack_name, attack_func in attacks:
            att_std = attack_func(np.copy(stego_std))
            try:
                ext_std = self.algo_standard.extract(att_std, expected_length=payload_chars)
                ber_std = Utils.calculate_ber(message, ext_std)
            except Exception as e:
                ber_std = 100.0

            att_clu = attack_func(np.copy(stego_clu))
            try:
                ext_clu = self.algo_clustered.extract(att_clu, expected_length=payload_chars)
                ber_clu = Utils.calculate_ber(message, ext_clu)
            except Exception as e:
                ber_clu = 100.0

            att_adp = attack_func(np.copy(stego_adp))
            try:
                ext_adp = self.algo_adaptive.extract(att_adp, expected_length=payload_chars)
                ber_adp = Utils.calculate_ber(message, ext_adp)
            except Exception as e:
                ber_adp = 100.0
            
            ber_dict = {'Std': ber_std, 'Clu': ber_clu, 'Adp': ber_adp}
            winner = min(ber_dict, key=ber_dict.get)
            
            self.results_robustness.append({
                'Attack': attack_name,
                'Std_BER': ber_std,
                'Clu_BER': ber_clu,
                'Adp_BER': ber_adp,
                'Winner': winner
            })
            
            print(f"{attack_name:<30} | {ber_std:>9.2f}% | {ber_clu:>9.2f}% | {ber_adp:>9.2f}% | {winner:<10}")

    def visualize_results(self):
        print("\n[Visualizing Results...]")
        
        # ✅ 1. PAYLOAD TEST
        if len(self.results_payload) > 0:
            df_payload = pd.DataFrame(self.results_payload)
            
            # Print capacity comparison dengan BITS
            print("\n" + "="*95)
            print("=== PERBANDINGAN KAPASITAS (Metrik Standar Akademik) ===")
            print("="*95)
            print(f"{'Method':<20} | {'Capacity (bits)':<18} | {'bpp':<10} | {'Chars':<12} | {'Relative':<10}")
            print("-" * 95)
            base_bits = self.capacity_std_bits
            for method, bits, bpp, chars in [
                ('Standard Edge', self.capacity_std_bits, self.bpp_std, self.capacity_std_chars),
                ('Clustered Edge', self.capacity_clu_bits, self.bpp_clu, self.capacity_clu_chars),
                ('Adaptive Edge', self.capacity_adp_bits, self.bpp_adp, self.capacity_adp_chars)
            ]:
                rel_pct = (bits / base_bits) * 100
                print(f"{method:<20} | {bits:>17,} | {bpp:>9.4f} | {chars:>11,} | {rel_pct:>9.1f}%")
            print("="*95)
            print("ℹ️  bpp = bits per pixel (normalisasi terhadap ukuran gambar)")
            print("ℹ️  Untuk publikasi, gunakan bits dan bpp sebagai metrik utama\n")
            
            # Grafik Payload
            plt.figure(figsize=(14, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(df_payload['Payload (%)'], df_payload['Std_PSNR'], 'r--o', label='Standard Edge')
            plt.plot(df_payload['Payload (%)'], df_payload['Clu_PSNR'], 'g-.^', label='Clustered Edge')
            plt.plot(df_payload['Payload (%)'], df_payload['Adp_PSNR'], 'b-s', label='Adaptive Edge')
            plt.title('Payload vs PSNR', fontsize=13, fontweight='bold')
            plt.xlabel('Payload Capacity (%)')
            plt.ylabel('PSNR (dB)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(df_payload['Payload (%)'], df_payload['Std_SSIM'], 'r--o', label='Standard Edge')
            plt.plot(df_payload['Payload (%)'], df_payload['Clu_SSIM'], 'g-.^', label='Clustered Edge')
            plt.plot(df_payload['Payload (%)'], df_payload['Adp_SSIM'], 'b-s', label='Adaptive Edge')
            plt.title('Payload vs SSIM', fontsize=13, fontweight='bold')
            plt.xlabel('Payload Capacity (%)')
            plt.ylabel('SSIM')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('result_payload_quality.png', dpi=150)
            print("   > Grafik disimpan: result_payload_quality.png")
        
        # ✅ 2. ROBUSTNESS TEST
        if len(self.results_robustness) > 0:
            df_rob = pd.DataFrame(self.results_robustness)
            
            plt.figure(figsize=(20, 8))
            x = np.arange(len(df_rob['Attack']))
            width = 0.25
            
            plt.bar(x - width, df_rob['Std_BER'], width, label='Standard Edge', color='red', alpha=0.7)
            plt.bar(x, df_rob['Clu_BER'], width, label='Clustered Edge', color='green', alpha=0.7)
            plt.bar(x + width, df_rob['Adp_BER'], width, label='Adaptive Edge', color='blue', alpha=0.7)
            
            plt.xlabel('Attack Type', fontsize=12)
            plt.ylabel('Bit Error Rate (%)', fontsize=12)
            plt.title('Comprehensive Robustness Test (Lower is Better)', fontsize=14, fontweight='bold')
            plt.xticks(x, df_rob['Attack'], rotation=60, ha='right', fontsize=9)
            plt.legend(fontsize=11)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 105)
            
            plt.tight_layout()
            plt.savefig('result_robustness_comprehensive.png', dpi=150)
            print("   > Grafik disimpan: result_robustness_comprehensive.png")
            
            # Print full results table
            print("\n" + "="*85)
            print("=== RINGKASAN HASIL UJI KETAHANAN ===")
            print("="*85)
            print(df_rob.to_string(index=False))
            
            # Winner statistics
            print("\n" + "="*50)
            print("=== STATISTIK PEMENANG ===")
            print("="*50)
            winner_counts = df_rob['Winner'].value_counts()
            for method, count in winner_counts.items():
                pct = (count / len(df_rob)) * 100
                print(f"{method:<15}: {count:>3} wins ({pct:>5.1f}%)")
            
            # Overall statistics dengan mapping yang BENAR
            print("\n" + "="*90)
            print("=== STATISTIK KESELURUHAN (Metrik Akademik) ===")
            print("="*90)
            print(f"{'Method':<20} | {'Cap (bits)':<12} | {'bpp':<10} | {'Avg BER':<10} | {'Win Rate':<10}")
            print("-" * 90)
            
            # GUNAKAN MAPPING DICT YANG BENAR
            method_mapping = {
                'Standard Edge': ('Std', self.capacity_std_bits, self.bpp_std),
                'Clustered Edge': ('Clu', self.capacity_clu_bits, self.bpp_clu),
                'Adaptive Edge': ('Adp', self.capacity_adp_bits, self.bpp_adp)
            }
            
            for method_name, (short_name, bits, bpp) in method_mapping.items():
                avg_ber = df_rob[f'{short_name}_BER'].mean()
                win_rate = (winner_counts.get(short_name, 0) / len(df_rob)) * 100
                print(f"{method_name:<20} | {bits:>11,} | {bpp:>9.4f} | {avg_ber:>9.2f}% | {win_rate:>9.1f}%")
            
            print("="*90)
            print("ℹ️  Metrik untuk paper: Capacity (bits), bpp, BER (%), PSNR (dB), SSIM")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print(f"Memulai Pengujian Otomatis pada: {TEST_IMAGE_PATH}")
    print(f"Konfigurasi: Threshold={EDGE_THRESHOLD}, Eps={ADP_EPS}, MinSamples={ADP_MIN_SAMPLES}")
    
    tester = StegoAutomatedTester(TEST_IMAGE_PATH)
    
    tester.run_payload_stress_test()
    tester.run_robustness_test()
    
    tester.visualize_results()
    
    print("\n✅ PENGUJIAN SELESAI. Silakan cek file PNG yang dihasilkan.")
