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
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

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
    def add_salt_pepper_noise(image, prob):
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    @staticmethod
    def apply_jpeg_compression(image, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg


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

    def run_robustness_test(self):
        print("\n[2/3] Menjalankan Robustness Test (Noise & Compression)...")
        
        # Fix payload 50%
        max_cap = self.algo_standard.get_capacity(self.original_image)
        message = Utils.generate_random_string(int(max_cap * 0.5))
        
        # Generate Base Stego Images
        _, stego_std = self.algo_standard.embed(self.original_image, message)
        _, stego_clu = self.algo_clustered.embed(self.original_image, message)
        _, stego_adp = self.algo_adaptive.embed(self.original_image, message)
        
        attacks = [
            ("No Attack", lambda x: x),
            ("Salt&Pepper (0.01)", lambda x: Utils.add_salt_pepper_noise(x, 0.01)),
            ("JPEG (Q=90)", lambda x: Utils.apply_jpeg_compression(x, 90)),
            ("JPEG (Q=70)", lambda x: Utils.apply_jpeg_compression(x, 70))
        ]
        
        for attack_name, attack_func in attacks:
            # Attack Standard
            att_std = attack_func(stego_std)
            # Real Extract Attempt
            try:
                ext_std = self.algo_standard.extract(att_std)
                ber_std = Utils.calculate_ber(message, ext_std)
            except:
                ber_std = 100.0 # Fail

            # Attack Clustered
            att_clu = attack_func(stego_clu)
            try:
                ext_clu = self.algo_clustered.extract(att_clu)
                ber_clu = Utils.calculate_ber(message, ext_clu)
            except:
                ber_clu = 100.0 # Fail

            # Attack Adaptive
            att_adp = attack_func(stego_adp)
            try:
                ext_adp = self.algo_adaptive.extract(att_adp)
                ber_adp = Utils.calculate_ber(message, ext_adp)
            except:
                ber_adp = 100.0 # Fail
            
            self.results_robustness.append({
                'Attack': attack_name,
                'Std_BER': ber_std,
                'Clu_BER': ber_clu,
                'Adp_BER': ber_adp
            })
            print(f"   Attack: {attack_name} | Std BER: {ber_std:.2f}% | Clu BER: {ber_clu:.2f}% | Adp BER: {ber_adp:.2f}%")

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
        
        plt.figure(figsize=(10, 5))
        x = np.arange(len(df_rob['Attack']))
        width = 0.25
        
        plt.bar(x - width, df_rob['Std_BER'], width, label='Standard Edge', color='red', alpha=0.7)
        plt.bar(x, df_rob['Clu_BER'], width, label='Clustered Edge', color='green', alpha=0.7)
        plt.bar(x + width, df_rob['Adp_BER'], width, label='Adaptive Edge', color='blue', alpha=0.7)
        
        plt.xlabel('Attack Type')
        plt.ylabel('Bit Error Rate (%)')
        plt.title('Robustness Comparison (Lower is Better)')
        plt.xticks(x, df_rob['Attack'])
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('result_robustness.png')
        print("   > Grafik disimpan: result_robustness.png")

        # 3. Print Data Tables
        print("\n=== RINGKASAN HASIL UJI KAPASITAS ===")
        print(df_payload.to_string(index=False))
        
        print("\n=== RINGKASAN HASIL UJI KETAHANAN ===")
        print(df_rob.to_string(index=False))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Menggunakan konstanta global
    print(f"Memulai Pengujian Otomatis pada: {TEST_IMAGE_PATH}")
    print(f"Konfigurasi Global: Th={EDGE_THRESHOLD}, Eps={ADP_EPS}, MinS={ADP_MIN_SAMPLES}")
    
    tester = StegoAutomatedTester(TEST_IMAGE_PATH)
    
    # Jalankan semua tes
    tester.run_payload_stress_test()
    tester.run_robustness_test()
    tester.run_parameter_sensitivity()
    
    # Visualisasi
    tester.visualize_results()
    
    print("\n✅ PENGUJIAN SELESAI. Silakan cek file PNG yang dihasilkan.")
