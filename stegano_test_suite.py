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
# Ubah parameter di sini untuk mengontrol seluruh pengujian

# Path Gambar
TEST_IMAGE_PATH = "images/pepper.tiff"

# Parameter Umum
EDGE_THRESHOLD = 30         # Ambang batas deteksi tepi (0-255)

# Parameter Adaptive (DBSCAN & Clustering)
ADP_EPS = 0.15                 # Jarak maksimum antar sampel
ADP_MIN_SAMPLES = 4         # Jumlah sampel minimum per cluster
ADP_USE_ISOLATED = False    # True: Isolated Pixels, False: Grouped Pixels
ADP_VAR_PERCENTILE = 75     # Persentil varians untuk threshold adaptif
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# BAGIAN 1: KELAS ALGORITMA (WRAPPER)
# =============================================================================

class StandardEdgeStego:
    """
    Wrapper untuk Metode Standard Edge-Based (REAL IMPLEMENTATION).
    """
    def get_capacity(self, image):
        # Save temp untuk dibaca fungsi asli
        cv2.imwrite("temp_cap_std.png", image)
        success, res = SteganographyEdge.get_capacity("temp_cap_std.png", threshold=EDGE_THRESHOLD)
        if success:
            return res['max_chars']
        return 0

    def embed(self, image, message):
        # Bridge: Numpy -> File -> Algo -> File -> Numpy
        cv2.imwrite("temp_src_std.png", image)
        output_path = "temp_out_std.png"
        
        # Hapus output lama jika ada
        if os.path.exists(output_path): os.remove(output_path)
            
        success, msg = SteganographyEdge.encode_message(
            "temp_src_std.png", message, output_path, threshold=EDGE_THRESHOLD
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image # Return original on fail

    def extract(self, stego_image):
        cv2.imwrite("temp_ext_std.png", stego_image)
        success, msg = SteganographyEdge.decode_message("temp_ext_std.png", threshold=EDGE_THRESHOLD)
        if success:
            return msg
        return ""

class ClusteredEdgeStego:
    """
    Wrapper untuk Metode Clustered Edge (REAL IMPLEMENTATION).
    """
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
        
        if os.path.exists(output_path): os.remove(output_path)
        
        success, msg = SteganographyEdgeClustered.encode_message(
            "temp_src_clu.png", message, output_path, 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image

    def extract(self, stego_image):
        cv2.imwrite("temp_ext_clu.png", stego_image)
        success, msg = SteganographyEdgeClustered.decode_message(
            "temp_ext_clu.png", self.threshold, self.eps, self.min_samples, 
            self.use_isolated
        )
        if success:
            return msg
        return ""

class AdaptiveEdgeStego:
    """
    Wrapper untuk Metode Adaptive Edge (REAL IMPLEMENTATION).
    """
    def __init__(self):
        # Menggunakan parameter GLOBAL dari atas
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
        
        if os.path.exists(output_path): os.remove(output_path)
        
        success, msg = SteganographyEdgeAdaptive.encode_message(
            "temp_src_adp.png", message, output_path, 
            self.threshold, self.eps, self.min_samples, 
            self.use_isolated, self.variance_percentile
        )
        
        if success and os.path.exists(output_path):
            return True, cv2.imread(output_path)
        return False, image

    def extract(self, stego_image):
        cv2.imwrite("temp_ext_adp.png", stego_image)
        success, msg = SteganographyEdgeAdaptive.decode_message(
            "temp_ext_adp.png", self.threshold, self.eps, self.min_samples, 
            self.use_isolated
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
        # Konversi ke biner
        bin_orig = Utils.str_to_bin(original_msg)
        bin_extr = Utils.str_to_bin(extracted_msg)
        
        len_orig = len(bin_orig)
        len_extr = len(bin_extr)
        
        if len_orig == 0: return 0.0
        if len_extr == 0: return 100.0
        
        # Hitung bit yang berbeda pada bagian yang overlap
        min_len = min(len_orig, len_extr)
        bit_errors = sum(c1 != c2 for c1, c2 in zip(bin_orig[:min_len], bin_extr[:min_len]))
        
        # Tambahkan penalti untuk sisa panjang yang tidak match
        length_diff = abs(len_orig - len_extr)
        total_errors = bit_errors + length_diff
        
        # Normalisasi agar max 100%
        # Kita bagi dengan max(len_orig, len_extr) agar tidak meledak > 100%
        ber = (total_errors / max(len_orig, len_extr)) * 100.0
        
        return min(ber, 100.0)

    @staticmethod
    # automated_test.py - class Utils  
    def add_salt_pepper_noise(image, prob):
            """
            Add salt & pepper noise (FIXED: Total noise = prob, not 2*prob)
            
            Args:
                image: Input image (numpy array)
                prob: Total noise probability (0.0-1.0)
                    Split 50-50 between salt and pepper
            
            Returns:
                Noisy image (numpy array)
            """
            output = np.copy(image)  # ← Gunakan copy, bukan zeros
            
            # Split noise probability
            prob_pepper = prob / 2  # Half for pepper (black)
            prob_salt = prob / 2    # Half for salt (white)
            thres_salt = 1 - prob_salt
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    
                    if rdn < prob_pepper:
                        # Pepper (black)
                        output[i][j] = [0, 0, 0]
                    elif rdn > thres_salt:
                        # Salt (white)
                        output[i][j] = [255, 255, 255]
                    # else: keep original (no change needed since we use copy)
            
            return output


    @staticmethod
    def apply_jpeg_compression(image, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        """
        Add Gaussian (Normal Distribution) noise
        Common in electronic circuit noise and thermal noise
        
        Args:
            image: Input image
            mean: Mean of Gaussian distribution (default: 0)
            sigma: Standard deviation (default: 25, range: 10-50 typical)
        
        Returns:
            Noisy image
        """
        output = np.copy(image).astype(np.float64)
        
        # Generate Gaussian noise
        gaussian = np.random.normal(mean, sigma, image.shape)
        
        # Add noise
        output = output + gaussian
        
        # Clip to valid range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    
    @staticmethod
    def add_rayleigh_noise(image, scale=25):
        """
        Add Rayleigh noise
        Common in ultrasound imaging and radar systems
        Models magnitude of noise with Gaussian components
        
        Args:
            image: Input image
            scale: Scale parameter (sigma), controls noise intensity
                  Typical range: 10-50
        
        Returns:
            Noisy image
        """
        output = np.copy(image).astype(np.float64)
        
        # Generate Rayleigh noise
        # Rayleigh distribution with scale parameter
        rayleigh = np.random.rayleigh(scale, image.shape)
        
        # Subtract mean to center noise around 0
        rayleigh = rayleigh - np.mean(rayleigh)
        
        # Add noise
        output = output + rayleigh
        
        # Clip to valid range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    
    @staticmethod
    def add_erlang_noise(image, shape=2, scale=15):
        """
        Add Erlang (Gamma) noise
        Special case of Gamma distribution (shape = integer)
        Models sum of exponential random variables
        Common in queueing theory and telecommunications
        
        Args:
            image: Input image
            shape: Shape parameter k (integer, default: 2)
                  Higher k = more Gaussian-like
            scale: Scale parameter (default: 15)
                  Controls noise intensity
        
        Returns:
            Noisy image
        """
        output = np.copy(image).astype(np.float64)
        
        # Generate Erlang (Gamma) noise
        # Erlang is Gamma with integer shape parameter
        erlang = np.random.gamma(shape, scale, image.shape)
        
        # Center noise around 0
        erlang = erlang - np.mean(erlang)
        
        # Add noise
        output = output + erlang
        
        # Clip to valid range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    
    @staticmethod
    def add_uniform_noise(image, low=-50, high=50):
        """
        Add Uniform noise
        All values in range [low, high] have equal probability
        Models quantization noise and digital round-off errors
        
        Args:
            image: Input image
            low: Lower bound of noise (default: -50)
            high: Upper bound of noise (default: 50)
        
        Returns:
            Noisy image
        """
        output = np.copy(image).astype(np.float64)
        
        # Generate Uniform noise
        uniform = np.random.uniform(low, high, image.shape)
        
        # Add noise
        output = output + uniform
        
        # Clip to valid range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    
    @staticmethod
    def add_exponential_noise(image, scale=30):
        """
        Add Exponential noise (BONUS)
        Models time between events in Poisson process
        
        Args:
            image: Input image
            scale: Scale parameter (1/lambda), controls noise intensity
        
        Returns:
            Noisy image
        """
        output = np.copy(image).astype(np.float64)
        
        # Generate Exponential noise
        exponential = np.random.exponential(scale, image.shape)
        
        # Center around 0
        exponential = exponential - np.mean(exponential)
        
        # Add noise
        output = output + exponential
        
        # Clip to valid range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output


# =============================================================================
# BAGIAN 3: AUTOMATED TESTING FRAMEWORK
# =============================================================================

class StegoAutomatedTester:
    def __init__(self, image_path):
        # Load Image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            # Generate dummy image if not found
            print("⚠ Gambar tidak ditemukan. Membuat dummy image gradient...")
            self.original_image = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(512):
                self.original_image[i, :] = (i % 256, (i*2) % 256, (255-i) % 256)
        
        self.algo_standard = StandardEdgeStego()
        self.algo_clustered = ClusteredEdgeStego()
        self.algo_adaptive = AdaptiveEdgeStego() # Uses GLOBAL CONFIG internally
        
        self.results_payload = []
        self.results_robustness = []
        self.results_sensitivity = []

    def run_payload_stress_test(self):
        print("\n[1/3] Menjalankan Payload Capacity Stress Test...")
        
        # Hitung baseline kapasitas (menggunakan standard sebagai acuan 100%)
        max_cap = self.algo_standard.get_capacity(self.original_image)
        
        payload_percentages = range(10, 101, 10) # 10% sampai 100%
        
        for pct in payload_percentages:
            msg_len = int(max_cap * (pct / 100.0))
            message = Utils.generate_random_string(msg_len)
            
            # --- Test Standard ---
            ret_s, stego_s = self.algo_standard.embed(self.original_image, message)
            if ret_s:
                p_s = psnr(self.original_image, stego_s)
                s_s = ssim(self.original_image, stego_s, channel_axis=2)
                # Real Extract
                ext_s = self.algo_standard.extract(stego_s)
                ber_s = Utils.calculate_ber(message, ext_s)
            else:
                p_s, s_s, ber_s = 0, 0, 100

            # --- Test Clustered ---
            ret_c, stego_c = self.algo_clustered.embed(self.original_image, message)
            if ret_c:
                p_c = psnr(self.original_image, stego_c)
                s_c = ssim(self.original_image, stego_c, channel_axis=2)
                # Real Extract
                ext_c = self.algo_clustered.extract(stego_c)
                ber_c = Utils.calculate_ber(message, ext_c)
            else:
                p_c, s_c, ber_c = 0, 0, 100

            # --- Test Adaptive ---
            ret_a, stego_a = self.algo_adaptive.embed(self.original_image, message)
            if ret_a:
                p_a = psnr(self.original_image, stego_a)
                s_a = ssim(self.original_image, stego_a, channel_axis=2)
                # Real Extract
                ext_a = self.algo_adaptive.extract(stego_a)
                ber_a = Utils.calculate_ber(message, ext_a)
            else:
                p_a, s_a, ber_a = 0, 0, 100

            self.results_payload.append({
                'Payload (%)': pct,
                'Std_PSNR': p_s, 'Std_SSIM': s_s,
                'Clu_PSNR': p_c, 'Clu_SSIM': s_c,
                'Adp_PSNR': p_a, 'Adp_SSIM': s_a
            })
            print(f"   Progress: {pct}% Payload | Std PSNR: {p_s:.2f} | Clu PSNR: {p_c:.2f} | Adp PSNR: {p_a:.2f}")

    # automated_test.py - class StegoAutomatedTester

    def run_robustness_test(self):
        print("\n[2/3] Menjalankan Robustness Test (Noise Distributions & Attacks)...")
        
        # Fix payload 50%
        max_cap = self.algo_standard.get_capacity(self.original_image)
        message = Utils.generate_random_string(int(max_cap * 0.5))
        
        # Generate Base Stego Images
        _, stego_std = self.algo_standard.embed(self.original_image, message)
        _, stego_clu = self.algo_clustered.embed(self.original_image, message)
        _, stego_adp = self.algo_adaptive.embed(self.original_image, message)
        
        # ✅ COMPREHENSIVE ATTACK SUITE
        attacks = [
            # Baseline
            ("No Attack", lambda x: x),
            
            # === NOISE DISTRIBUTIONS (Statistical) ===
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
            
            # === IMPULSE NOISE ===
            ("Salt&Pepper (0.5%)", lambda x: Utils.add_salt_pepper_noise(x, 0.005)),
            ("Salt&Pepper (1.0%)", lambda x: Utils.add_salt_pepper_noise(x, 0.01)),
            ("Salt&Pepper (2.0%)", lambda x: Utils.add_salt_pepper_noise(x, 0.02)),
        
            
            # === COMPRESSION ATTACKS ===
            ("JPEG (Q=90)", lambda x: Utils.apply_jpeg_compression(x, 90)),
            ("JPEG (Q=70)", lambda x: Utils.apply_jpeg_compression(x, 70)),
            ("JPEG (Q=50)", lambda x: Utils.apply_jpeg_compression(x, 50)),
        ]
        
        print(f"\n{'Attack Type':<30} | {'Std BER':<10} | {'Clu BER':<10} | {'Adp BER':<10} | {'Winner':<10}")
        print("-" * 85)
        
        for attack_name, attack_func in attacks:
            # Attack Standard
            att_std = attack_func(np.copy(stego_std))
            try:
                ext_std = self.algo_standard.extract(att_std)
                ber_std = Utils.calculate_ber(message, ext_std)
            except:
                ber_std = 100.0

            # Attack Clustered
            att_clu = attack_func(np.copy(stego_clu))
            try:
                ext_clu = self.algo_clustered.extract(att_clu)
                ber_clu = Utils.calculate_ber(message, ext_clu)
            except:
                ber_clu = 100.0

            # Attack Adaptive
            att_adp = attack_func(np.copy(stego_adp))
            try:
                ext_adp = self.algo_adaptive.extract(att_adp)
                ber_adp = Utils.calculate_ber(message, ext_adp)
            except:
                ber_adp = 100.0
            
            # Determine winner (lowest BER)
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

    def run_parameter_sensitivity(self):
        print("\n[3/3] Menjalankan Parameter Sensitivity Test (Adaptive Only)...")
        
        # Note: Test ini sengaja memvariasikan parameter untuk mencari yang terbaik,
        # mengabaikan GLOBAL CONFIG untuk EPS dan MIN_SAMPLES.
        
        eps_values = [3, 5, 7, 9]
        min_samples_values = [3, 5, 10]
        
        message = Utils.generate_random_string(1000) # Fixed small message
        
        for eps in eps_values:
            for ms in min_samples_values:
                # Create instance with specific params for sensitivity test
                algo = AdaptiveEdgeStego()
                algo.eps = eps
                algo.min_samples = ms
                
                # Real Embed call
                ret, stego = algo.embed(self.original_image, message)
                
                if ret:
                    p_val = psnr(self.original_image, stego)
                    # Get detected edges from capacity info
                    cap = algo.get_capacity(self.original_image)
                    edges_detected = cap # Rough proxy
                else:
                    p_val = 0
                    edges_detected = 0
                    
                self.results_sensitivity.append({
                    'EPS': eps,
                    'Min_Samples': ms,
                    'PSNR': p_val,
                    'Detected_Edges': edges_detected
                })
        print("   Parameter sensitivity selesai.")

    def visualize_results(self):
        print("\n[Visualizing Results...]")
        
        # 1. Grafik Payload vs Quality (PSNR & SSIM)
        df_payload = pd.DataFrame(self.results_payload)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: PSNR
        plt.subplot(1, 2, 1)
        plt.plot(df_payload['Payload (%)'], df_payload['Std_PSNR'], 'r--o', label='Standard Edge')
        plt.plot(df_payload['Payload (%)'], df_payload['Clu_PSNR'], 'g-.^', label='Clustered Edge')
        plt.plot(df_payload['Payload (%)'], df_payload['Adp_PSNR'], 'b-s', label='Adaptive Edge')
        plt.title('Comparison: Payload vs PSNR')
        plt.xlabel('Payload Capacity (%)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: SSIM
        plt.subplot(1, 2, 2)
        plt.plot(df_payload['Payload (%)'], df_payload['Std_SSIM'], 'r--o', label='Standard Edge')
        plt.plot(df_payload['Payload (%)'], df_payload['Clu_SSIM'], 'g-.^', label='Clustered Edge')
        plt.plot(df_payload['Payload (%)'], df_payload['Adp_SSIM'], 'b-s', label='Adaptive Edge')
        plt.title('Comparison: Payload vs SSIM')
        plt.xlabel('Payload Capacity (%)')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('result_payload_quality.png')
        print("   > Grafik disimpan: result_payload_quality.png")
        
        # 2. Grafik Robustness (Bar Chart)
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
        
        
        # ✅ 3. Grafik Khusus: NOISE DISTRIBUTIONS ONLY
        plt.figure(figsize=(14, 6))
        
        noise_attacks = df_rob[
            df_rob['Attack'].str.contains('Gaussian|Rayleigh|Erlang|Uniform|Exponential', case=False)
        ]
        
        if len(noise_attacks) > 0:
            x_noise = np.arange(len(noise_attacks))
            
            plt.bar(x_noise - width, noise_attacks['Std_BER'], width, 
                    label='Standard Edge', color='red', alpha=0.7)
            plt.bar(x_noise, noise_attacks['Clu_BER'], width, 
                    label='Clustered Edge', color='green', alpha=0.7)
            plt.bar(x_noise + width, noise_attacks['Adp_BER'], width, 
                    label='Adaptive Edge', color='blue', alpha=0.7)
            
            plt.xlabel('Noise Distribution Type', fontsize=12)
            plt.ylabel('Bit Error Rate (%)', fontsize=12)
            plt.title('Robustness Against Statistical Noise Distributions', fontsize=14, fontweight='bold')
            plt.xticks(x_noise, noise_attacks['Attack'], rotation=45, ha='right')
            plt.legend(fontsize=11)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 105)
            
            plt.tight_layout()
            plt.savefig('result_robustness_noise_distributions.png', dpi=150)
            print("   > Grafik noise distributions disimpan: result_robustness_noise_distributions.png")
        
        
        # ✅ 4. Grafik Grouped by Attack Category (Updated)
        plt.figure(figsize=(12, 6))
        
        # Categorize attacks
        noise_dist = df_rob[df_rob['Attack'].str.contains('Gaussian|Rayleigh|Erlang|Uniform|Exponential')]
        impulse_noise = df_rob[df_rob['Attack'].str.contains('Salt|Speckle')]
        filtering = df_rob[df_rob['Attack'].str.contains('Median|Blur|Sharp')]
        compression = df_rob[df_rob['Attack'].str.contains('JPEG')]
        geometric = df_rob[df_rob['Attack'].str.contains('Scaling|Rotation|Crop')]
        
        categories = ['Noise\nDistributions', 'Impulse\nNoise', 'Filtering', 'Compression', 'Geometric']
        std_means = [
            noise_dist['Std_BER'].mean() if len(noise_dist) > 0 else 0,
            impulse_noise['Std_BER'].mean() if len(impulse_noise) > 0 else 0,
            filtering['Std_BER'].mean() if len(filtering) > 0 else 0,
            compression['Std_BER'].mean() if len(compression) > 0 else 0,
            geometric['Std_BER'].mean() if len(geometric) > 0 else 0
        ]
        clu_means = [
            noise_dist['Clu_BER'].mean() if len(noise_dist) > 0 else 0,
            impulse_noise['Clu_BER'].mean() if len(impulse_noise) > 0 else 0,
            filtering['Clu_BER'].mean() if len(filtering) > 0 else 0,
            compression['Clu_BER'].mean() if len(compression) > 0 else 0,
            geometric['Clu_BER'].mean() if len(geometric) > 0 else 0
        ]
        adp_means = [
            noise_dist['Adp_BER'].mean() if len(noise_dist) > 0 else 0,
            impulse_noise['Adp_BER'].mean() if len(impulse_noise) > 0 else 0,
            filtering['Adp_BER'].mean() if len(filtering) > 0 else 0,
            compression['Adp_BER'].mean() if len(compression) > 0 else 0,
            geometric['Adp_BER'].mean() if len(geometric) > 0 else 0
        ]
        
        x_cat = np.arange(len(categories))
        plt.bar(x_cat - width, std_means, width, label='Standard Edge', color='red', alpha=0.7)
        plt.bar(x_cat, clu_means, width, label='Clustered Edge', color='green', alpha=0.7)
        plt.bar(x_cat + width, adp_means, width, label='Adaptive Edge', color='blue', alpha=0.7)
        
        plt.xlabel('Attack Category', fontsize=12)
        plt.ylabel('Average BER (%)', fontsize=12)
        plt.title('Average Robustness by Attack Category', fontsize=14, fontweight='bold')
        plt.xticks(x_cat, categories)
        plt.legend(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(max(std_means), max(clu_means), max(adp_means)) * 1.1)
        
        plt.tight_layout()
        plt.savefig('result_robustness_by_category.png', dpi=150)
        print("   > Grafik kategori disimpan: result_robustness_by_category.png")
        
        
        # Print tables
        print("\n" + "="*85)
        print("=== RINGKASAN HASIL UJI KETAHANAN (COMPREHENSIVE) ===")
        print("="*85)
        print(df_rob.to_string(index=False))
        
        # ✅ Winner Statistics
        print("\n" + "="*50)
        print("=== STATISTIK PEMENANG (Metode Terbaik per Attack) ===")
        print("="*50)
        winner_counts = df_rob['Winner'].value_counts()
        for method, count in winner_counts.items():
            pct = (count / len(df_rob)) * 100
            print(f"{method:<15}: {count:>3} wins ({pct:>5.1f}%)")
        
        # ✅ Summary Statistics
        print("\n" + "="*70)
        print("=== STATISTIK RATA-RATA BER PER KATEGORI ===")
        print("="*70)
        print(f"{'Category':<20} | {'Standard':<10} | {'Clustered':<10} | {'Adaptive':<10}")
        print("-" * 70)
        for i, cat in enumerate(categories):
            print(f"{cat.replace(chr(10), ' '):<20} | {std_means[i]:>9.2f}% | {clu_means[i]:>9.2f}% | {adp_means[i]:>9.2f}%")
        
        # Overall average
        print("-" * 70)
        print(f"{'OVERALL AVERAGE':<20} | {df_rob['Std_BER'].mean():>9.2f}% | "
            f"{df_rob['Clu_BER'].mean():>9.2f}% | {df_rob['Adp_BER'].mean():>9.2f}%")
        print("="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Menggunakan konstanta global
    print(f"Memulai Pengujian Otomatis pada: {TEST_IMAGE_PATH}")
    print(f"Konfigurasi Global: Th={EDGE_THRESHOLD}, Eps={ADP_EPS}, MinS={ADP_MIN_SAMPLES}")
    
    tester = StegoAutomatedTester(TEST_IMAGE_PATH)
    
    # Jalankan semua tes
    # tester.run_payload_stress_test()
    tester.run_robustness_test()
    
    # Visualisasi
    tester.visualize_results()
    
    print("\n✅ PENGUJIAN SELESAI. Silakan cek file PNG yang dihasilkan.")
