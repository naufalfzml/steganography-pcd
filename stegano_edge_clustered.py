from PIL import Image
from edge_clustering import EdgeClustering


class SteganographyEdgeClustered:
    """
    Steganography berbasis edge dengan DBSCAN clustering
    Optimasi: Hanya gunakan edge pixels yang ter-cluster atau isolated
    """

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50,
                       eps=3, min_samples=5, use_noise=False):
        """
        Encode pesan menggunakan edge pixels yang sudah di-clustering

        Args:
            image_path: Path gambar input
            message: Pesan yang akan disembunyikan
            output_path: Path output
            threshold: Threshold edge detection
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            use_noise: True = gunakan noise pixels, False = gunakan clustered pixels

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
                image_path, threshold, eps, min_samples, use_noise
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            # Encode
            encoded_img = img.copy()

            # Tambah delimiter
            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            # Validasi kapasitas
            max_bits = len(edge_coords) * 3
            if len(message_bits) > max_bits:
                max_chars = max_bits // 8 - 7
                return False, f"Pesan terlalu panjang! Maksimal {max_chars} karakter."

            # Embed ke edge pixels
            data_index = 0
            for x, y in edge_coords:
                if data_index >= len(message_bits):
                    break

                pixel = list(img.getpixel((x, y)))

                # Modifikasi LSB
                for i in range(3):
                    if data_index < len(message_bits):
                        pixel[i] = pixel[i] & ~1 | int(message_bits[data_index])
                        data_index += 1

                encoded_img.putpixel((x, y), tuple(pixel))

            # Save
            encoded_img.save(output_path)

            cluster_type = "noise (isolated)" if use_noise else "clustered (grouped)"
            return True, f"Pesan berhasil disembunyikan di {data_index // 3} {cluster_type} edge pixels! Gambar disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path, threshold=50, eps=3, min_samples=5, use_noise=False):
        """
        Decode pesan dari edge pixels yang sudah di-clustering

        Args:
            image_path: Path gambar dengan pesan
            threshold: Threshold edge detection (HARUS SAMA dengan encode)
            eps: DBSCAN eps (HARUS SAMA dengan encode)
            min_samples: DBSCAN min_samples (HARUS SAMA dengan encode)
            use_noise: HARUS SAMA dengan encode

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
                image_path, threshold, eps, min_samples, use_noise
            )

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge pixels yang sesuai kriteria"

            # Ekstrak bit
            message_bits = []

            for x, y in edge_coords:
                pixel = img.getpixel((x, y))

                # Ambil LSB
                for value in pixel:
                    message_bits.append(str(value & 1))

            # Convert ke karakter
            message = ""
            for i in range(0, len(message_bits), 8):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    char = chr(int(''.join(byte), 2))
                    message += char

                    # Cek delimiter
                    if message.endswith("<<END>>"):
                        return True, message[:-7]

            return True, message if message else "Tidak ada pesan ditemukan!"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_capacity(image_path, threshold=50, eps=3, min_samples=5, use_noise=False):
        """
        Hitung kapasitas dengan clustering

        Args:
            image_path: Path gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples
            use_noise: True = noise pixels, False = clustered pixels

        Returns:
            tuple: (success: bool, result: dict)
        """
        return EdgeClustering.get_capacity_clustered(
            image_path, threshold, eps, min_samples, use_noise
        )
