from PIL import Image
from edge_detection import EdgeDetection


class SteganographyEdge:
    # Steganography pada area edge gambar menggunakan Sobel

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=50):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, f"Tidak ada edge ditemukan dengan threshold {threshold}. Coba turunkan threshold."

            encoded_img = img.copy()

            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            max_bits = len(edge_coords) * 3
            if len(message_bits) > max_bits:
                max_chars = max_bits // 8 - 7
                return False, f"Pesan terlalu panjang! Maksimal {max_chars} karakter untuk edge pixels."

            data_index = 0
            for x, y in edge_coords:
                if data_index >= len(message_bits):
                    break

                pixel = list(img.getpixel((x, y)))

                for i in range(3):
                    if data_index < len(message_bits):
                        pixel[i] = pixel[i] & ~1 | int(message_bits[data_index])
                        data_index += 1

                encoded_img.putpixel((x, y), tuple(pixel))

            encoded_img.save(output_path)

            return True, f"Pesan berhasil disembunyikan di {data_index // 3} edge pixels! Gambar disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path, threshold=50, expected_length=None):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, "Tidak ada edge ditemukan. Pesan mungkin menggunakan threshold yang berbeda."

            message_bits = []

            for x, y in edge_coords:
                pixel = img.getpixel((x, y))

                for value in pixel:
                    message_bits.append(str(value & 1))

            message = ""
            
            # Jika expected_length ditentukan
            if expected_length is not None:
                target_bits = expected_length * 8
                bits_to_process = message_bits[:target_bits]
                
                for i in range(0, len(bits_to_process), 8):
                    byte = bits_to_process[i:i+8]
                    if len(byte) == 8:
                        char = chr(int(''.join(byte), 2))
                        message += char
                return True, message

            # Normal decoding dengan delimiter
            for i in range(0, len(message_bits), 8):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    char = chr(int(''.join(byte), 2))
                    message += char

                    if message.endswith("<<END>>"):
                        return True, message[:-7]

            return True, message if message else "Tidak ada pesan ditemukan!"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"


    @staticmethod
    def get_capacity(image_path, threshold=50):
        success, stats = EdgeDetection.get_edge_statistics(image_path, threshold)

        if not success:
            return False, stats

        return True, {
            'max_chars': stats['max_capacity_chars'],
            'edge_pixels': stats['edge_pixels'],
            'edge_percentage': stats['edge_percentage'],
            'threshold': threshold
        }