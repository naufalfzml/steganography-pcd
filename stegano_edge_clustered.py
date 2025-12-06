from PIL import Image
from edge_clustering import EdgeClustering


class SteganographyEdgeClustered:
    # Steganography berbasis edge dengan DBSCAN clustering

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50,
                       eps=6, min_samples=20, use_isolated=False):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            encoded_img = img.copy()

            message_bits_data = ''.join([format(ord(char), '08b') for char in message])
            
            message_length = len(message)
            length_bits = format(message_length, '032b') 
            message_bits = length_bits + message_bits_data

            max_bits = len(edge_coords) * 3
            if len(message_bits) > max_bits:
                max_chars = (max_bits - 32) // 8
                return False, f"Pesan terlalu panjang! Maksimal {max_chars} karakter."

            data_index = 0
            pixels_modified = 0
            
            for x, y in edge_coords:
                if data_index >= len(message_bits):
                    break

                pixel = list(img.getpixel((x, y)))

                for i in range(3):
                    if data_index < len(message_bits):
                        pixel[i] = (pixel[i] & ~1) | int(message_bits[data_index])
                        data_index += 1

                encoded_img.putpixel((x, y), tuple(pixel))
                pixels_modified += 1

            encoded_img.save(output_path)

            pixel_type = "isolated" if use_isolated else "grouped"
            return True, (
                f"Pesan berhasil disembunyikan!\n"
                f"- Pixels modified: {pixels_modified} {pixel_type} edge pixels\n"
                f"- Message length: {message_length} characters\n"
                f"- Bits embedded: {len(message_bits)} bits\n"
                f"- Output: {output_path}"
            )

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path, threshold=50, eps=6, min_samples=20, use_isolated=False, expected_length=None):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            message_bits = []

            for x, y in edge_coords:
                pixel = img.getpixel((x, y))

                for value in pixel:
                    message_bits.append(str(value & 1))
            
            # Mode robustness test
            if expected_length is not None:
                if len(message_bits) < 16:
                    return False, "Tidak cukup data"
                
                message_bits_data = message_bits[16:]
                target_bits = expected_length * 8
                
                bits_to_process = message_bits_data[:target_bits]
                
                message = ""
                for i in range(0, len(bits_to_process), 8):
                    byte = bits_to_process[i:i+8]
                    if len(byte) == 8:
                        char = chr(int(''.join(byte), 2))
                        message += char
                return True, message

            # Normal decoding
            if len(message_bits) < 32:
                return False, "Tidak cukup data untuk decode"

            length_bits = ''.join(message_bits[:32])
            message_length = int(length_bits, 2)

            if message_length == 0:
                return False, "Tidak ada pesan ditemukan (length = 0)"

            max_possible_chars = (len(message_bits) - 32) // 8
            if message_length > max_possible_chars:
                message_length = max_possible_chars

            message_bits_data = message_bits[32:32 + (message_length * 8)]

            message = ""
            for i in range(0, len(message_bits_data), 8):
                byte = message_bits_data[i:i+8]
                if len(byte) == 8:
                    char = chr(int(''.join(byte), 2))
                    message += char

            return True, message

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"


    @staticmethod
    def get_capacity(image_path, threshold=50, eps=6, min_samples=20, use_isolated=False):
        return EdgeClustering.get_capacity_clustered(
            image_path, threshold, eps, min_samples, use_isolated
        )

    @staticmethod
    def compare_capacity(image_path, threshold=50, eps=6, min_samples=20):
        try:
            success_grouped, cap_grouped = SteganographyEdgeClustered.get_capacity(
                image_path, threshold, eps, min_samples, use_isolated=False
            )

            if not success_grouped:
                return False, cap_grouped

            success_isolated, cap_isolated = SteganographyEdgeClustered.get_capacity(
                image_path, threshold, eps, min_samples, use_isolated=True
            )

            if not success_isolated:
                return False, cap_isolated

            comparison = {
                'grouped_edges': {
                    'pixels': cap_grouped['edge_pixels'],
                    'percentage': cap_grouped['edge_percentage'],
                    'max_chars': cap_grouped['max_chars']
                },
                'isolated_edges': {
                    'pixels': cap_isolated['edge_pixels'],
                    'percentage': cap_isolated['edge_percentage'],
                    'max_chars': cap_isolated['max_chars']
                },
                'difference': {
                    'pixels': cap_grouped['edge_pixels'] - cap_isolated['edge_pixels'],
                    'max_chars': cap_grouped['max_chars'] - cap_isolated['max_chars']
                },
                'recommendation': 'Use grouped edges for higher capacity' if cap_grouped['max_chars'] > cap_isolated['max_chars'] * 2 else 'Use isolated edges for better security'
            }

            return True, comparison

        except Exception as e:
            return False, f"Error: {str(e)}"