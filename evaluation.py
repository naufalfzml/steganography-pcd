import numpy as np
from PIL import Image
import math
import random


class Evaluation:
    # Evaluasi kualitas steganography - PSNR dan BER

    @staticmethod
    def calculate_mse(image1_path, image2_path):
        try:
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')

            arr1 = np.array(img1, dtype=np.float64)
            arr2 = np.array(img2, dtype=np.float64)

            if arr1.shape != arr2.shape:
                return False, "Dimensi gambar tidak sama!"

            mse = np.mean((arr1 - arr2) ** 2)

            return True, mse

        except FileNotFoundError as e:
            return False, f"File tidak ditemukan: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def calculate_psnr(image1_path, image2_path):
        success, mse = Evaluation.calculate_mse(image1_path, image2_path)

        if not success:
            return False, mse

        if mse == 0:
            return True, float('inf')

        max_pixel = 255.0
        psnr = 10 * math.log10((max_pixel ** 2) / mse)

        return True, psnr

    @staticmethod
    def calculate_ber(original_message, decoded_message):
        # BER = Jumlah bit berbeda / Total bit pesan asli
        try:
            original_bits = ''.join([format(ord(char), '08b') for char in original_message])
            decoded_bits = ''.join([format(ord(char), '08b') for char in decoded_message])
            
            original_len = len(original_bits)
            decoded_len = len(decoded_bits)
            
            if original_len == 0:
                return True, {
                    'total_bits': 0,
                    'error_bits': 0,
                    'correct_bits': 0,
                    'ber': 0.0,
                    'ber_percentage': 0.0,
                    'accuracy': 100.0
                }
            
            error_count = 0
            
            for i in range(original_len):
                if i >= decoded_len:
                    error_count += 1
                elif original_bits[i] != decoded_bits[i]:
                    error_count += 1
            
            ber = error_count / original_len
            correct_bits = original_len - error_count
            
            result = {
                'total_bits': original_len,
                'error_bits': error_count,
                'correct_bits': correct_bits,
                'ber': ber,
                'ber_percentage': ber * 100,
                'accuracy': (1 - ber) * 100,
                'decoded_length': decoded_len,
                'length_diff': decoded_len - original_len
            }
            
            return True, result
            
        except Exception as e:
            return False, f"Error: {str(e)}"


    @staticmethod
    def interpret_psnr(psnr):
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
            return "Poor (>= 10% error)"

    @staticmethod
    def add_salt_and_pepper_noise(image_path, output_path, amount=0.01, random_seed=None):
        try:
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)
                
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            noisy_img = np.copy(img_array)
            
            prob_pepper = amount / 2
            prob_salt = amount / 2
            thres_pepper = prob_pepper
            thres_salt = 1 - prob_salt
            
            height, width = img_array.shape[:2]
            
            for i in range(height):
                for j in range(width):
                    rdn = random.random()
                    
                    if rdn < thres_pepper:
                        noisy_img[i, j] = [0, 0, 0]
                    elif rdn > thres_salt:
                        noisy_img[i, j] = [255, 255, 255]
            
            Image.fromarray(noisy_img.astype('uint8')).save(output_path)
            
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
