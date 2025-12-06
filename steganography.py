from PIL import Image


class Steganography:
    # Steganography pada gambar menggunakan metode LSB

    @staticmethod
    def encode_message(image_path, message, output_path):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            encoded_img = img.copy()
            width, height = img.size

            message += "<<END>>"
            message_bits = ''.join([format(ord(char), '08b') for char in message])

            max_bits = width * height * 3
            if len(message_bits) > max_bits:
                return False, "Pesan terlalu panjang untuk gambar ini!"

            data_index = 0
            for y in range(height):
                for x in range(width):
                    if data_index < len(message_bits):
                        pixel = list(img.getpixel((x, y)))

                        for i in range(3):
                            if data_index < len(message_bits):
                                pixel[i] = pixel[i] & ~1 | int(message_bits[data_index])
                                data_index += 1

                        encoded_img.putpixel((x, y), tuple(pixel))
                    else:
                        break
                if data_index >= len(message_bits):
                    break

            encoded_img.save(output_path)
            return True, f"Pesan berhasil disembunyikan! Gambar disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def decode_message(image_path):
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            message_bits = []

            for y in range(height):
                for x in range(width):
                    pixel = img.getpixel((x, y))

                    for value in pixel:
                        message_bits.append(str(value & 1))

            message = ""
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
    def get_image_capacity(image_path):
        try:
            img = Image.open(image_path)
            width, height = img.size

            max_bits = width * height * 3
            max_chars = max_bits // 8

            # Kurangi untuk delimiter
            max_chars -= 7

            return True, max_chars

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"