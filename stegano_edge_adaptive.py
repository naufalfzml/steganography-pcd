from PIL import Image
import numpy as np
from edge_clustering import EdgeClustering


class SteganographyEdgeAdaptive:
    """
    Steganography dengan Adaptive Embedding berdasarkan cluster characteristics

    Key Difference:
    - Standard: Embed di semua edge pixels dengan intensitas sama
    - Adaptive: Embed lebih banyak bit di edge dengan gradient tinggi (cluster besar)
                Embed lebih sedikit bit di edge dengan gradient rendah (noise/cluster kecil)

    Benefit:
    - PSNR lebih tinggi karena perubahan di area dengan variasi tinggi
    - BER lebih rendah karena embedding lebih robust di area stabil
    """

    @staticmethod
    def calculate_edge_strength(image, x, y):
        """
        Hitung kekuatan edge di sekitar pixel (x, y)
        Digunakan untuk menentukan berapa bit yang bisa di-embed

        Args:
            image: PIL Image (RGB)
            x, y: Koordinat pixel

        Returns:
            float: Edge strength (0-255)
        """
        # Convert ke grayscale array
        gray = image.convert('L')
        arr = np.array(gray)
        height, width = arr.shape

        # Ambil neighborhood 3x3
        y_start = max(0, y - 1)
        y_end = min(height, y + 2)
        x_start = max(0, x - 1)
        x_end = min(width, x + 2)

        neighborhood = arr[y_start:y_end, x_start:x_end]

        # Hitung variance sebagai ukuran edge strength
        # Variance tinggi = edge kuat = bisa embed lebih banyak bit
        variance = np.var(neighborhood)

        return variance

    @staticmethod
    def adaptive_embed_bits(pixel_value, message_bits, bit_index, strength, mode='adaptive'):
        """
        Embed bits secara adaptive berdasarkan edge strength

        Args:
            pixel_value: Nilai pixel original (0-255)
            message_bits: String bit message
            bit_index: Index bit saat ini
            strength: Edge strength (variance)
            mode: 'adaptive' atau 'standard'

        Returns:
            tuple: (new_pixel_value, bits_embedded, new_bit_index)
        """
        if mode == 'standard':
            # Standard LSB: Hanya 1 bit di LSB
            if bit_index < len(message_bits):
                new_value = pixel_value & ~1 | int(message_bits[bit_index])
                return new_value, 1, bit_index + 1
            return pixel_value, 0, bit_index

        elif mode == 'adaptive':
            # Adaptive: Embed 1-2 bit tergantung strength
            # Strength tinggi (>500) = embed 2 bit
            # Strength rendah (<500) = embed 1 bit (LSB only)

            if strength > 500:  # High variance area
                # Embed 2 bit: LSB dan bit ke-2
                bits_to_embed = min(2, len(message_bits) - bit_index)

                if bits_to_embed == 2:
                    # Modify 2 LSB
                    new_value = pixel_value & ~3  # Clear 2 LSB
                    new_value |= int(message_bits[bit_index:bit_index+2], 2)
                    return new_value, 2, bit_index + 2
                elif bits_to_embed == 1:
                    # Hanya 1 bit tersisa
                    new_value = pixel_value & ~1 | int(message_bits[bit_index])
                    return new_value, 1, bit_index + 1
                else:
                    return pixel_value, 0, bit_index
            else:  # Low variance area
                # Embed 1 bit: LSB only (safer)
                if bit_index < len(message_bits):
                    new_value = pixel_value & ~1 | int(message_bits[bit_index])
                    return new_value, 1, bit_index + 1
                return pixel_value, 0, bit_index

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50,
                       eps=3, min_samples=5, use_adaptive=True):
        """
        Encode dengan adaptive embedding

        Args:
            image_path: Path gambar
            message: Pesan
            output_path: Path output
            threshold: Threshold edge detection
            eps, min_samples: DBSCAN parameters
            use_adaptive: True = adaptive embedding, False = standard LSB

        Returns:
            tuple: (success, message)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan clustered edge pixels
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_noise=False
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai"

            # Prepare message
            encoded_img = img.copy()
            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            # Encode dengan mode yang dipilih
            mode = 'adaptive' if use_adaptive else 'standard'
            bit_index = 0
            total_bits_embedded = 0
            pixels_used = 0

            for x, y in edge_coords:
                if bit_index >= len(message_bits):
                    break

                # Hitung edge strength
                strength = SteganographyEdgeAdaptive.calculate_edge_strength(img, x, y)

                pixel = list(img.getpixel((x, y)))

                # Embed ke setiap channel RGB
                for i in range(3):
                    if bit_index >= len(message_bits):
                        break

                    new_value, bits_embedded, bit_index = SteganographyEdgeAdaptive.adaptive_embed_bits(
                        pixel[i], message_bits, bit_index, strength, mode
                    )
                    pixel[i] = new_value
                    total_bits_embedded += bits_embedded

                encoded_img.putpixel((x, y), tuple(pixel))
                pixels_used += 1

            # Save
            encoded_img.save(output_path)

            method = "Adaptive (variable bit)" if use_adaptive else "Standard (1 bit LSB)"
            return True, f"Pesan berhasil disembunyikan dengan {method}! Total {total_bits_embedded} bit di {pixels_used} pixels. Gambar: {output_path}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path, threshold=50, eps=3, min_samples=5, use_adaptive=True):
        """
        Decode dengan adaptive extraction

        Args:
            image_path: Path gambar
            threshold, eps, min_samples: HARUS SAMA dengan encode
            use_adaptive: HARUS SAMA dengan encode

        Returns:
            tuple: (success, message)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan edge pixels (sama urutan dengan encode)
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_noise=False
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels"

            # Decode
            mode = 'adaptive' if use_adaptive else 'standard'
            message_bits = []

            for x, y in edge_coords:
                # Hitung edge strength (sama dengan encode)
                strength = SteganographyEdgeAdaptive.calculate_edge_strength(img, x, y)

                pixel = img.getpixel((x, y))

                for value in pixel:
                    if mode == 'standard':
                        # Extract 1 bit (LSB)
                        message_bits.append(str(value & 1))
                    elif mode == 'adaptive':
                        if strength > 500:
                            # Extract 2 bit
                            message_bits.append(str((value >> 1) & 1))  # Bit ke-2
                            message_bits.append(str(value & 1))         # LSB
                        else:
                            # Extract 1 bit (LSB)
                            message_bits.append(str(value & 1))

            # Convert bits ke karakter
            message = ""
            for i in range(0, len(message_bits), 8):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    char = chr(int(''.join(byte), 2))
                    message += char

                    if message.endswith("<<END>>"):
                        return True, message[:-7]

            return True, message if message else "Tidak ada pesan"

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_capacity(image_path, threshold=50, eps=3, min_samples=5, use_adaptive=True):
        """
        Estimate kapasitas dengan adaptive embedding

        Adaptive embedding bisa menyimpan lebih banyak bit di area dengan variance tinggi
        """
        try:
            img = Image.open(image_path)

            # Dapatkan clustered pixels
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_noise=False
            )

            if not success:
                return False, edge_coords

            # Estimate kapasitas
            if use_adaptive:
                # Hitung rata-rata strength
                total_strength = 0
                for x, y in edge_coords[:min(100, len(edge_coords))]:  # Sample 100 pixels
                    strength = SteganographyEdgeAdaptive.calculate_edge_strength(img, x, y)
                    total_strength += strength

                avg_strength = total_strength / min(100, len(edge_coords))

                # Estimate bits per pixel
                if avg_strength > 500:
                    bits_per_channel = 1.8  # Mostly 2 bit
                else:
                    bits_per_channel = 1.2  # Mostly 1 bit

                max_bits = int(len(edge_coords) * 3 * bits_per_channel)
            else:
                # Standard: 3 bit per pixel
                max_bits = len(edge_coords) * 3

            max_chars = max_bits // 8 - 7

            return True, {
                'edge_pixels': len(edge_coords),
                'max_chars': max_chars,
                'mode': 'Adaptive' if use_adaptive else 'Standard',
                'estimated_bits': max_bits
            }

        except Exception as e:
            return False, f"Error: {str(e)}"
