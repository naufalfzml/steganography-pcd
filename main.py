from PIL import Image
import os

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
                raise ValueError("Pesan terlalu panjang untuk gambar ini!")

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
            print(f"✓ Pesan berhasil disembunyikan!")
            print(f"✓ Gambar disimpan di: {output_path}")

        except FileNotFoundError:
            print(f"✗ Error: File '{image_path}' tidak ditemukan!")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

    @staticmethod
    def decode_message(image_path):
        """
        Mengekstrak pesan dari gambar

        Args:
            image_path: Path ke gambar yang berisi pesan tersembunyi

        Returns:
            Pesan yang berhasil diekstrak
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
                        return message[:-7]  # Hapus delimiter

            return message if message else "Tidak ada pesan ditemukan!"

        except FileNotFoundError:
            print(f"✗ Error: File '{image_path}' tidak ditemukan!")
            return None
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None


def print_header():
    """Menampilkan header program"""
    print("\n" + "="*50)
    print("    PROGRAM STEGANOGRAPHY GAMBAR")
    print("    Metode: LSB (Least Significant Bit)")
    print("="*50)


def main():
    """Fungsi utama program"""
    stego = Steganography()

    while True:
        print_header()
        print("\nMenu:")
        print("1. Sembunyikan pesan dalam gambar (Encode)")
        print("2. Ekstrak pesan dari gambar (Decode)")
        print("3. Keluar")
        print("-"*50)

        choice = input("Pilih menu (1-3): ").strip()

        if choice == '1':
            print("\n--- ENCODE: Sembunyikan Pesan ---")
            image_path = input("Path gambar input: ").strip()

            if not os.path.exists(image_path):
                print(f"✗ File '{image_path}' tidak ditemukan!")
                continue

            message = input("Masukkan pesan yang ingin disembunyikan: ")
            output_path = input("Path untuk menyimpan gambar hasil (contoh: output.png): ").strip()

            print("\nMemproses...")
            stego.encode_message(image_path, message, output_path)

        elif choice == '2':
            print("\n--- DECODE: Ekstrak Pesan ---")
            image_path = input("Path gambar yang berisi pesan tersembunyi: ").strip()

            if not os.path.exists(image_path):
                print(f"✗ File '{image_path}' tidak ditemukan!")
                continue

            print("\nMemproses...")
            message = stego.decode_message(image_path)

            if message:
                print("\n" + "="*50)
                print("Pesan yang ditemukan:")
                print("-"*50)
                print(message)
                print("="*50)

        elif choice == '3':
            print("\nTerima kasih telah menggunakan program ini!")
            break

        else:
            print("\n✗ Pilihan tidak valid! Silakan pilih 1-3.")

        input("\nTekan Enter untuk melanjutkan...")


if __name__ == "__main__":
    main()
