from PIL import Image
from edge_clustering import EdgeClustering


class SteganographyEdgeClustered:
    """
    Steganography berbasis edge dengan DBSCAN clustering
    Optimasi: Hanya gunakan edge pixels yang ter-cluster atau isolated
    """

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50,
                       eps=6, min_samples=20, use_isolated=False):
        """
        Encode pesan menggunakan edge pixels yang sudah di-clustering

        Args:
            image_path: Path gambar input
            message: Pesan yang akan disembunyikan
            output_path: Path output
            threshold: Threshold edge detection
            eps: DBSCAN eps parameter (default: 6)
            min_samples: DBSCAN min_samples parameter (default: 20)
            use_isolated: True = gunakan isolated pixels (lebih aman), 
                         False = gunakan grouped pixels (kapasitas lebih besar)

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan optimized edge pixels
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            # Encode dengan length prefix
            encoded_img = img.copy()

            # Convert message ke bits
            message_bits_data = ''.join([format(ord(char), '08b') for char in message])
            
            # Tambahkan length prefix (16 bits untuk length)
            message_length = len(message)
            if message_length > 65535:  # Max untuk 16 bits
                return False, "Pesan terlalu panjang (maksimal 65535 karakter)"
            
            length_bits = format(message_length, '016b')
            message_bits = length_bits + message_bits_data

            # Validasi kapasitas
            max_bits = len(edge_coords) * 3
            if len(message_bits) > max_bits:
                max_chars = (max_bits - 16) // 8
                return False, f"Pesan terlalu panjang! Maksimal {max_chars} karakter."

            # Embed ke edge pixels
            data_index = 0
            pixels_modified = 0
            
            for x, y in edge_coords:
                if data_index >= len(message_bits):
                    break

                pixel = list(img.getpixel((x, y)))

                # Modifikasi LSB untuk RGB channels
                for i in range(3):
                    if data_index < len(message_bits):
                        # Clear LSB dan set dengan bit pesan
                        pixel[i] = (pixel[i] & ~1) | int(message_bits[data_index])
                        data_index += 1

                encoded_img.putpixel((x, y), tuple(pixel))
                pixels_modified += 1

            # Save
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
    def decode_message(image_path, threshold=50, eps=6, min_samples=20, use_isolated=False):
        """
        Decode pesan dari edge pixels yang sudah di-clustering

        Args:
            image_path: Path gambar dengan pesan
            threshold: Threshold edge detection (HARUS SAMA dengan encode)
            eps: DBSCAN eps (HARUS SAMA dengan encode)
            min_samples: DBSCAN min_samples (HARUS SAMA dengan encode)
            use_isolated: HARUS SAMA dengan encode

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan optimized edge pixels (HARUS SAMA URUTAN dengan encode!)
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            # Ekstrak bit dari edge pixels
            message_bits = []

            for x, y in edge_coords:
                pixel = img.getpixel((x, y))

                # Ambil LSB dari setiap channel RGB
                for value in pixel:
                    message_bits.append(str(value & 1))

            # Minimal harus ada 16 bits untuk length
            if len(message_bits) < 16:
                return False, "Tidak cukup data untuk decode"

            # Ekstrak length dari 16 bits pertama
            length_bits = ''.join(message_bits[:16])
            message_length = int(length_bits, 2)

            # Validasi length
            if message_length == 0:
                return False, "Tidak ada pesan ditemukan (length = 0)"

            max_possible_chars = (len(message_bits) - 16) // 8
            if message_length > max_possible_chars:
                return False, f"Length tidak valid ({message_length} > {max_possible_chars})"

            # Ekstrak message bits
            message_bits_data = message_bits[16:16 + (message_length * 8)]

            # Convert bits ke karakter
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
        """
        Hitung kapasitas dengan clustering

        Args:
            image_path: Path gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples
            use_isolated: True = isolated pixels, False = grouped pixels

        Returns:
            tuple: (success: bool, result: dict)
        """
        return EdgeClustering.get_capacity_clustered(
            image_path, threshold, eps, min_samples, use_isolated
        )

    @staticmethod
    def compare_capacity(image_path, threshold=50, eps=6, min_samples=20):
        """
        Bandingkan kapasitas antara isolated vs grouped edge pixels

        Args:
            image_path: Path gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples

        Returns:
            tuple: (success: bool, comparison: dict)
        """
        try:
            # Kapasitas untuk grouped edges
            success_grouped, cap_grouped = SteganographyEdgeClustered.get_capacity(
                image_path, threshold, eps, min_samples, use_isolated=False
            )

            if not success_grouped:
                return False, cap_grouped

            # Kapasitas untuk isolated edges
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
