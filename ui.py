import os
from steganography import Steganography


def print_header():
    """Menampilkan header program"""
    print("\n" + "="*50)
    print("    PROGRAM STEGANOGRAPHY GAMBAR")
    print("    Metode: LSB (Least Significant Bit)")
    print("="*50)


def print_success(message):
    """Menampilkan pesan sukses"""
    print(f"✓ {message}")


def print_error(message):
    """Menampilkan pesan error"""
    print(f"✗ {message}")


def encode_menu():
    """Menu untuk encode (sembunyikan pesan)"""
    print("\n--- ENCODE: Sembunyikan Pesan ---")

    # Input path gambar
    image_path = input("Path gambar input: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Cek kapasitas gambar
    success, result = Steganography.get_image_capacity(image_path)
    if success:
        print(f"ℹ Kapasitas maksimal: {result} karakter")

    # Input pesan
    message = input("Masukkan pesan yang ingin disembunyikan: ")

    if success and len(message) > result:
        print_error(f"Pesan terlalu panjang! Maksimal {result} karakter, Anda memasukkan {len(message)} karakter.")
        return

    # Input output path
    output_path = input("Path untuk menyimpan gambar hasil (contoh: output.png): ").strip()

    # Proses encode
    print("\nMemproses...")
    success, msg = Steganography.encode_message(image_path, message, output_path)

    if success:
        print_success(msg)
    else:
        print_error(msg)


def decode_menu():
    """Menu untuk decode (ekstrak pesan)"""
    print("\n--- DECODE: Ekstrak Pesan ---")

    # Input path gambar
    image_path = input("Path gambar yang berisi pesan tersembunyi: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Proses decode
    print("\nMemproses...")
    success, message = Steganography.decode_message(image_path)

    if success:
        print("\n" + "="*50)
        print("Pesan yang ditemukan:")
        print("-"*50)
        print(message)
        print("="*50)
    else:
        print_error(message)


def capacity_menu():
    """Menu untuk cek kapasitas gambar"""
    print("\n--- CEK KAPASITAS GAMBAR ---")

    # Input path gambar
    image_path = input("Path gambar: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Cek kapasitas
    success, result = Steganography.get_image_capacity(image_path)

    if success:
        print("\n" + "="*50)
        print(f"Kapasitas maksimal: {result} karakter")
        print(f"Atau sekitar: {result // 100} kalimat pendek")
        print("="*50)
    else:
        print_error(result)


def display_menu():
    """Menampilkan menu utama"""
    print("\nMenu:")
    print("1. Sembunyikan pesan dalam gambar (Encode)")
    print("2. Ekstrak pesan dari gambar (Decode)")
    print("3. Cek kapasitas gambar")
    print("4. Keluar")
    print("-"*50)


def run():
    """Fungsi utama untuk menjalankan program"""
    while True:
        print_header()
        display_menu()

        choice = input("Pilih menu (1-4): ").strip()

        if choice == '1':
            encode_menu()
        elif choice == '2':
            decode_menu()
        elif choice == '3':
            capacity_menu()
        elif choice == '4':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        else:
            print_error("Pilihan tidak valid! Silakan pilih 1-4.")

        input("\nTekan Enter untuk melanjutkan...")
