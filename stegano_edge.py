from PIL import Image
from edge_detection import EdgeDetection


class SteganographyEdge:
    """
    Kelas untuk melakukan steganography pada area edge gambar menggunakan Sobel edge detection
    Lebih aman karena perubahan di edge tidak terlihat oleh mata manusia
    """

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50):
        """
        Menyembunyikan pesan hanya di pixel-pixel edge gambar

        Args:
            image_path: Path ke gambar input
            message: Pesan yang akan disembunyikan
            output_path: Path untuk menyimpan gambar hasil
            threshold: Threshold untuk edge detection (default: 50)

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke RGB jika belum
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan koordinat edge pixels
            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, f"Tidak ada edge ditemukan dengan threshold {threshold}. Coba turunkan threshold."

            # Encode pesan
            encoded_img = img.copy()

            # Tambahkan delimiter
            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            # Cek apakah pesan terlalu panjang
            max_bits = len(edge_coords) * 3  # 3 bit per pixel (RGB)
            if len(message_bits) > max_bits:
                max_chars = max_bits // 8 - 7
                return False, f"Pesan terlalu panjang! Maksimal {max_chars} karakter untuk edge pixels."

            # Sembunyikan pesan hanya di edge pixels
            data_index = 0
            for x, y in edge_coords:
                if data_index >= len(message_bits):
                    break

                pixel = list(img.getpixel((x, y)))

                # Modifikasi LSB dari setiap channel RGB
                for i in range(3):
                    if data_index < len(message_bits):
                        pixel[i] = pixel[i] & ~1 | int(message_bits[data_index])
                        data_index += 1

                encoded_img.putpixel((x, y), tuple(pixel))

            # Simpan gambar hasil
            encoded_img.save(output_path)

            return True, f"Pesan berhasil disembunyikan di {data_index // 3} edge pixels! Gambar disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path, threshold=50):
        """
        Mengekstrak pesan dari edge pixels gambar

        Args:
            image_path: Path ke gambar yang berisi pesan tersembunyi
            threshold: Threshold untuk edge detection (harus sama dengan saat encode)

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke RGB jika belum
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Dapatkan koordinat edge pixels (harus sama urutan dengan saat encode)
            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge ditemukan. Pesan mungkin menggunakan threshold yang berbeda."

            # Ekstrak bit dari edge pixels
            message_bits = []

            for x, y in edge_coords:
                pixel = img.getpixel((x, y))

                # Ambil LSB dari setiap channel RGB
                for value in pixel:
                    message_bits.append(str(value & 1))

            # Convert bits ke karakter
            message = ""
            for i in range(0, len(message_bits), 8):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    char = chr(int(''.join(byte), 2))
                    message += char

                    # Cek apakah sudah mencapai delimiter
                    if message.endswith("<<END>>"):
                        return True, message[:-7]  # Hapus delimiter

            return True, message if message else "Tidak ada pesan ditemukan!"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_capacity(image_path, threshold=50):
        """
        Menghitung kapasitas maksimal pesan untuk edge-based steganography

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection

        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        success, stats = EdgeDetection.get_edge_statistics(image_path, threshold)

        if not success:
            return False, stats

        return True, {
            'max_chars': stats['max_capacity_chars'],
            'edge_pixels': stats['edge_pixels'],
            'edge_percentage': stats['edge_percentage'],
            'threshold': threshold
        }