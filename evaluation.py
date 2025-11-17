import numpy as np
from PIL import Image
import math


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
        Hitung Bit Error Rate (BER)
        BER = Jumlah bit berbeda / Total bit
        BER rendah = decode lebih akurat

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

            # Pad ke panjang yang sama (untuk handle decode incomplete)
            max_len = max(len(original_bits), len(decoded_bits))
            original_bits = original_bits.ljust(max_len, '0')
            decoded_bits = decoded_bits.ljust(max_len, '0')

            # Hitung bit errors
            error_count = 0
            for i in range(max_len):
                if original_bits[i] != decoded_bits[i]:
                    error_count += 1

            # BER
            ber = error_count / max_len if max_len > 0 else 0

            result = {
                'total_bits': max_len,
                'error_bits': error_count,
                'correct_bits': max_len - error_count,
                'ber': ber,
                'ber_percentage': ber * 100,
                'accuracy': (1 - ber) * 100
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

