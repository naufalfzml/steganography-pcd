from PIL import Image


class Steganography:
    """
    Kelas untuk melakukan steganography pada gambar menggunakan metode LSB (Least Significant Bit)
    """

    @staticmethod
    def encode_message(image_path, message, output_path):
        """
        Menyembunyikan pesan dalam gambar

        Args:
            image_path: Path ke gambar input
            message: Pesan yang akan disembunyikan
            output_path: Path untuk menyimpan gambar hasil

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke RGB jika belum
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Encode pesan
            encoded_img = img.copy()
            width, height = img.size

            # Tambahkan delimiter untuk menandai akhir pesan
            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            # Cek apakah pesan terlalu panjang untuk gambar
            max_bits = width * height * 3
            if len(message_bits) > max_bits:
                return False, "Pesan terlalu panjang untuk gambar ini!"

            # Sembunyikan pesan ke dalam pixel
            data_index = 0
            for y in range(height):
                for x in range(width):
                    if data_index < len(message_bits):
                        pixel = list(img.getpixel((x, y)))

                        # Modifikasi bit terakhir (LSB) dari setiap channel RGB
                        for i in range(3):
                            if data_index < len(message_bits):
                                pixel[i] = pixel[i] & ~1 | int(message_bits[data_index])
                                data_index += 1

                        encoded_img.putpixel((x, y), tuple(pixel))
                    else:
                        break
                if data_index >= len(message_bits):
                    break

            # Simpan gambar hasil
            encoded_img.save(output_path)
            return True, f"Pesan berhasil disembunyikan! Gambar disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path):
        """
        Mengekstrak pesan dari gambar

        Args:
            image_path: Path ke gambar yang berisi pesan tersembunyi

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke RGB jika belum
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            message_bits = []

            # Ekstrak bit dari pixel
            for y in range(height):
                for x in range(width):
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
    def get_image_capacity(image_path):
        """
        Menghitung kapasitas maksimal pesan yang bisa disimpan dalam gambar

        Args:
            image_path: Path ke gambar

        Returns:
            tuple: (success: bool, capacity: int atau error_message: str)
        """
        try:
            img = Image.open(image_path)
            width, height = img.size

            # Setiap pixel RGB bisa menyimpan 3 bit
            # Setiap karakter membutuhkan 8 bit
            max_bits = width * height * 3
            max_chars = max_bits // 8

            # Kurangi untuk delimiter "<<END>>"
            max_chars -= 7

            return True, max_chars

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"