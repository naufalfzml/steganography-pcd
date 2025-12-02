import numpy as np
from PIL import Image
import math
import random


class Evaluation:
    """
    Kelas untuk evaluasi kualitas steganography
    Metrics: PSNR (Peak Signal-to-Noise Ratio) dan BER (Bit Error Rate)
    """

    @staticmethod
    def calculate_mse(image1_path, image2_path):
        """
        Hitung Mean Squared Error antara dua gambar

        Args:
            image1_path: Path gambar original
            image2_path: Path gambar stego

        Returns:
            tuple: (success: bool, mse: float atau error message)
        """
        try:
            # Buka kedua gambar
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')

            # Convert ke array
            arr1 = np.array(img1, dtype=np.float64)
            arr2 = np.array(img2, dtype=np.float64)

            # Cek dimensi sama
            if arr1.shape != arr2.shape:
                return False, "Dimensi gambar tidak sama!"

            # Hitung MSE
            mse = np.mean((arr1 - arr2) ** 2)

            return True, mse

        except FileNotFoundError as e:
            return False, f"File tidak ditemukan: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def calculate_psnr(image1_path, image2_path):
        """
        Hitung PSNR (Peak Signal-to-Noise Ratio)
        PSNR tinggi = kualitas lebih baik (perubahan minimal)

        Formula: PSNR = 10 * log10(MAX^2 / MSE)
        MAX = 255 untuk gambar 8-bit

        Args:
            image1_path: Path gambar original
            image2_path: Path gambar stego

        Returns:
            tuple: (success: bool, psnr: float atau error message)
        """
        success, mse = Evaluation.calculate_mse(image1_path, image2_path)

        if not success:
            return False, mse

        # Jika MSE = 0, gambar identik
        if mse == 0:
            return True, float('inf')  # PSNR infinite

        # Hitung PSNR
        max_pixel = 255.0
        psnr = 10 * math.log10((max_pixel ** 2) / mse)

        return True, psnr

    @staticmethod
    def calculate_ber(original_message, decoded_message):
        """
        Hitung Bit Error Rate (BER) berdasarkan panjang pesan ASLI
        BER = Jumlah bit berbeda / Total bit pesan asli
        BER rendah = decode lebih akurat
        
        Logika:
        - Bit yang hilang (decoded lebih pendek) = dihitung sebagai ERROR
        - Bit yang berlebih (decoded lebih panjang) = DIABAIKAN (trim)
        - Comparison dilakukan sepanjang pesan asli
        
        Args:
            original_message: Pesan asli
            decoded_message: Pesan hasil decode
        
        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        try:
            # Convert ke binary
            original_bits = ''.join([format(ord(char), '08b') for char in original_message])
            decoded_bits = ''.join([format(ord(char), '08b') for char in decoded_message])
            
            # Panjang pesan asli adalah ground truth
            original_len = len(original_bits)
            decoded_len = len(decoded_bits)
            
            # Edge case: pesan asli kosong
            if original_len == 0:
                return True, {
                    'total_bits': 0,
                    'error_bits': 0,
                    'correct_bits': 0,
                    'ber': 0.0,
                    'ber_percentage': 0.0,
                    'accuracy': 100.0
                }
            
            # Hitung bit errors
            error_count = 0
            
            # Bandingkan bit sepanjang pesan asli
            for i in range(original_len):
                if i >= decoded_len:
                    # Bit hilang (decoded lebih pendek) = ERROR
                    error_count += 1
                elif original_bits[i] != decoded_bits[i]:
                    # Bit berbeda = ERROR
                    error_count += 1
                # else: bit sama = BENAR
            
            # BER dihitung dari panjang ASLI
            ber = error_count / original_len
            correct_bits = original_len - error_count
            
            result = {
                'total_bits': original_len,
                'error_bits': error_count,
                'correct_bits': correct_bits,
                'ber': ber,
                'ber_percentage': ber * 100,
                'accuracy': (1 - ber) * 100,
                'decoded_length': decoded_len,  # Informasi tambahan
                'length_diff': decoded_len - original_len  # Informasi tambahan
            }
            
            return True, result
            
        except Exception as e:
            return False, f"Error: {str(e)}"


    @staticmethod
    def interpret_psnr(psnr):
        """
        Interpretasi nilai PSNR

        Args:
            psnr: Nilai PSNR dalam dB

        Returns:
            str: Interpretasi kualitas
        """
        if psnr == float('inf'):
            return "Perfect (Identical)"
        elif psnr >= 50:
            return "Excellent (Imperceptible)"
        elif psnr >= 40:
            return "Very Good (Almost Imperceptible)"
        elif psnr >= 30:
            return "Good (Perceptible but acceptable)"
        elif psnr >= 20:
            return "Fair (Noticeable degradation)"
        else:
            return "Poor (Significant degradation)"

    @staticmethod
    def interpret_ber(ber):
        """
        Interpretasi nilai BER

        Args:
            ber: Bit Error Rate (0-1)

        Returns:
            str: Interpretasi akurasi
        """
        if ber == 0:
            return "Perfect (No errors)"
        elif ber < 0.001:
            return "Excellent (< 0.1% error)"
        elif ber < 0.01:
            return "Very Good (< 1% error)"
        elif ber < 0.05:
            return "Good (< 5% error)"
        elif ber < 0.1:
            return "Fair (< 10% error)"
        else:
            return "Poor (≥ 10% error)"

    @staticmethod
    def add_salt_and_pepper_noise(image_path, output_path, amount=0.01, random_seed=None):
        """
        Menambahkan Salt and Pepper noise ke gambar (PER-PIXEL, REALISTIS)
        
        Args:
            image_path: Path gambar input
            output_path: Path gambar output
            amount: Persentase noise (0.0 - 1.0), default 0.01 (1%)
                   Ini adalah probabilitas per-pixel
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)
            # Buka gambar
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            # Copy array untuk dimodifikasi
            noisy_img = np.copy(img_array)
            
            # Threshold untuk salt (white) dan pepper (black)
            # amount/2 untuk salt, amount/2 untuk pepper
            prob_pepper = amount / 2  # Probabilitas pixel jadi hitam
            prob_salt = amount / 2     # Probabilitas pixel jadi putih
            thres_pepper = prob_pepper
            thres_salt = 1 - prob_salt
            
            # Iterasi per pixel (HEIGHT x WIDTH)
            # Semua channel RGB berubah bersamaan (per-pixel, bukan per-channel)
            height, width = img_array.shape[:2]
            
            for i in range(height):
                for j in range(width):
                    rdn = random.random()
                    
                    if rdn < thres_pepper:
                        # Pepper: Pixel jadi hitam (semua channel = 0)
                        noisy_img[i, j] = [0, 0, 0]
                    elif rdn > thres_salt:
                        # Salt: Pixel jadi putih (semua channel = 255)
                        noisy_img[i, j] = [255, 255, 255]
                    # else: tetap original (tidak diubah)
            
            # Simpan gambar
            Image.fromarray(noisy_img.astype('uint8')).save(output_path)
            
            # Hitung statistik noise yang ditambahkan
            total_pixels = height * width
            pepper_count = np.sum(np.all(noisy_img == [0, 0, 0], axis=2))
            salt_count = np.sum(np.all(noisy_img == [255, 255, 255], axis=2))
            actual_noise_pct = (pepper_count + salt_count) / total_pixels * 100
            
            return True, (f"Berhasil menambahkan noise | "
                         f"Salt: {salt_count} pixels, "
                         f"Pepper: {pepper_count} pixels "
                         f"({actual_noise_pct:.2f}% dari total)")
            
        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan"
        except Exception as e:
            return False, f"Gagal menambahkan noise: {str(e)}"
