import os
from steganography import Steganography
from stegano_edge import SteganographyEdge
from stegano_edge_adaptive import SteganographyEdgeAdaptive
from edge_detection import EdgeDetection
from edge_clustering import EdgeClustering
from evaluation import Evaluation


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


def encode_edge_menu():
    """Menu untuk encode menggunakan edge-based steganography"""
    print("\n--- ENCODE (EDGE-BASED): Sembunyikan Pesan di Edge ---")

    # Input path gambar
    image_path = input("Path gambar input: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Input threshold
    threshold_input = input("Threshold edge detection (default: 50, range: 0-255): ").strip()
    threshold = 50
    if threshold_input:
        try:
            threshold = int(threshold_input)
            if threshold < 0 or threshold > 255:
                print_error("Threshold harus antara 0-255. Menggunakan default: 50")
                threshold = 50
        except ValueError:
            print_error("Input tidak valid. Menggunakan default: 50")

    # Cek kapasitas edge
    print("\nMenganalisis edge gambar...")
    success, result = SteganographyEdge.get_capacity(image_path, threshold)

    if success:
        print(f"ℹ Edge pixels: {result['edge_pixels']} ({result['edge_percentage']:.2f}% dari total)")
        print(f"ℹ Kapasitas maksimal: {result['max_chars']} karakter")
    else:
        print_error(result)
        return

    # Input pesan
    message = input("\nMasukkan pesan yang ingin disembunyikan: ")

    if success and len(message) > result['max_chars']:
        print_error(f"Pesan terlalu panjang! Maksimal {result['max_chars']} karakter.")
        return

    # Input output path
    output_path = input("Path untuk menyimpan gambar hasil (contoh: output_edge.png): ").strip()

    # Proses encode
    print("\nMemproses...")
    success, msg = SteganographyEdge.encode_message(image_path, message, output_path, threshold)

    if success:
        print_success(msg)
        print(f"ℹ Threshold yang digunakan: {threshold} (INGAT ini untuk decode!)")
    else:
        print_error(msg)


def decode_edge_menu():
    """Menu untuk decode edge-based steganography"""
    print("\n--- DECODE (EDGE-BASED): Ekstrak Pesan dari Edge ---")

    # Input path gambar
    image_path = input("Path gambar yang berisi pesan tersembunyi: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Input threshold (harus sama dengan saat encode!)
    threshold_input = input("Threshold edge detection (harus sama dengan saat encode, default: 50): ").strip()
    threshold = 50
    if threshold_input:
        try:
            threshold = int(threshold_input)
            if threshold < 0 or threshold > 255:
                print_error("Threshold harus antara 0-255. Menggunakan default: 50")
                threshold = 50
        except ValueError:
            print_error("Input tidak valid. Menggunakan default: 50")

    # Proses decode
    print("\nMemproses...")
    success, message = SteganographyEdge.decode_message(image_path, threshold)

    if success:
        print("\n" + "="*50)
        print("Pesan yang ditemukan:")
        print("-"*50)
        print(message)
        print("="*50)
    else:
        print_error(message)


def visualize_edge_menu():
    """Menu untuk visualisasi edge detection"""
    print("\n--- VISUALISASI EDGE DETECTION ---")

    # Input path gambar
    image_path = input("Path gambar: ").strip()

    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    # Input threshold
    threshold_input = input("Threshold edge detection (default: 50): ").strip()
    threshold = 50
    if threshold_input:
        try:
            threshold = int(threshold_input)
            if threshold < 0 or threshold > 255:
                print_error("Threshold harus antara 0-255. Menggunakan default: 50")
                threshold = 50
        except ValueError:
            print_error("Input tidak valid. Menggunakan default: 50")

    # Input output path
    output_path = input("Path untuk menyimpan visualisasi edge (contoh: edge_visual.png): ").strip()

    # Proses visualisasi
    print("\nMemproses...")
    success, msg = EdgeDetection.visualize_edges(image_path, output_path, threshold)

    if success:
        print_success(msg)

        # Tampilkan statistik
        success_stats, stats = EdgeDetection.get_edge_statistics(image_path, threshold)
        if success_stats:
            print(f"\nStatistik Edge:")
            print(f"- Total pixels: {stats['total_pixels']}")
            print(f"- Edge pixels: {stats['edge_pixels']} ({stats['edge_percentage']:.2f}%)")
            print(f"- Kapasitas steganography: {stats['max_capacity_chars']} karakter")
    else:
        print_error(msg)


def cluster_visualization_menu():
    """Menu visualisasi clustering"""
    print("\n--- VISUALISASI CLUSTERING EDGE ---")

    image_path = input("Path gambar: ").strip()
    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    threshold = int(input("Threshold edge detection (default: 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default: 3): ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples (default: 5): ").strip() or 5)
    output_path = input("Path output visualisasi: ").strip()

    print("\nMemproses clustering...")
    success, msg = EdgeClustering.visualize_clusters(
        image_path, output_path, threshold, eps, min_samples
    )

    if success:
        print_success(msg)

        # Tampilkan statistik
        success_stat, result = EdgeClustering.cluster_edge_pixels(
            image_path, threshold, eps, min_samples
        )
        if success_stat:
            print(f"\nStatistik Clustering:")
            print(f"- Total edge pixels: {result['total_edge_pixels']}")
            print(f"- Jumlah cluster: {result['n_clusters']}")
            print(f"- Clustered pixels: {result['clustered_pixels']}")
            print(f"- Noise pixels: {result['noise_pixels']}")
            print(f"\nTop 5 cluster terbesar:")
            sorted_clusters = sorted(result['cluster_stats'].items(), key=lambda x: x[1], reverse=True)
            for i, (label, size) in enumerate(sorted_clusters[:5]):
                print(f"  Cluster {label}: {size} pixels")
    else:
        print_error(msg)


def encode_clustered_menu():
    """Menu encode dengan clustering"""
    print("\n--- ENCODE (CLUSTERED EDGE): Sembunyikan Pesan di Clustered Edge ---")

    image_path = input("Path gambar input: ").strip()
    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    threshold = int(input("Threshold edge detection (default: 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default: 3): ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples (default: 5): ").strip() or 5)

    print("\n  Pilih tipe edge pixels:")
    print("  1. Clustered pixels (grouped edges)")
    print("  2. Noise pixels (isolated edges)")
    choice = input("  Pilihan (1/2): ").strip()
    use_noise = (choice == '2')

    print("\nMenganalisis edge dan clustering...")
    success, result = SteganographyEdgeAdaptive.get_capacity(
        image_path, threshold, eps, min_samples, use_noise
    )

    if success:
        pixel_type = "noise (isolated)" if use_noise else "clustered (grouped)"
        print(f"ℹ Edge pixels ({pixel_type}): {result['edge_pixels']} ({result['edge_percentage']:.2f}%)")
        print(f"ℹ Kapasitas maksimal: {result['max_chars']} karakter")
    else:
        print_error(result)
        return

    message = input("\nMasukkan pesan: ")

    if len(message) > result['max_chars']:
        print_error(f"Pesan terlalu panjang! Maksimal {result['max_chars']} karakter.")
        return

    output_path = input("Path output: ").strip()

    print("\nMemproses...")
    success, msg = SteganographyEdgeAdaptive.encode_message(
        image_path, message, output_path, threshold, eps, min_samples, use_noise
    )

    if success:
        print_success(msg)
        print(f"ℹ Parameter: threshold={threshold}, eps={eps}, min_samples={min_samples}, use_noise={use_noise}")
        print("  INGAT parameter ini untuk decode!")
    else:
        print_error(msg)


def decode_clustered_menu():
    """Menu decode dengan clustering"""
    print("\n--- DECODE (CLUSTERED EDGE): Ekstrak Pesan dari Clustered Edge ---")

    image_path = input("Path gambar: ").strip()
    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    print("\nMasukkan parameter (HARUS SAMA dengan saat encode):")
    threshold = int(input("Threshold (default: 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default: 3): ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples (default: 5): ").strip() or 5)

    print("\n  Tipe edge pixels saat encode:")
    print("  1. Clustered pixels")
    print("  2. Noise pixels")
    choice = input("  Pilihan (1/2): ").strip()
    use_noise = (choice == '2')

    print("\nMemproses...")
    success, message = SteganographyEdgeAdaptive.decode_message(
        image_path, threshold, eps, min_samples, use_noise
    )

    if success:
        print("\n" + "="*50)
        print("Pesan yang ditemukan:")
        print("-"*50)
        print(message)
        print("="*50)
    else:
        print_error(message)


def compare_methods_menu():
    """Menu perbandingan metode"""
    print("\n--- PERBANDINGAN: Edge Tanpa Clustering vs Dengan Clustering ---")
    print("\nProses ini akan:")
    print("1. Encode pesan dengan edge (no clustering)")
    print("2. Encode pesan dengan edge (with clustering)")
    print("3. Decode kedua hasil")
    print("4. Hitung PSNR dan BER")
    print("5. Bandingkan hasil\n")

    original_image = input("Path gambar original: ").strip()
    if not os.path.exists(original_image):
        print_error(f"File tidak ditemukan!")
        return

    message = input("Pesan yang akan di-test: ")
    threshold = int(input("Threshold (default: 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default: 3): ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples (default: 5): ").strip() or 5)

    # Temporary files
    stego_no_cluster = "temp_stego_no_cluster.png"
    stego_with_cluster = "temp_stego_with_cluster.png"

    print("\n--- Step 1: Encode tanpa clustering ---")
    success1, msg1 = SteganographyEdge.encode_message(
        original_image, message, stego_no_cluster, threshold
    )
    if not success1:
        print_error(f"Encode no cluster gagal: {msg1}")
        return
    print_success("Encode tanpa clustering berhasil")

    print("\n--- Step 2: Encode dengan clustering ---")
    success2, msg2 = SteganographyEdgeAdaptive.encode_message(
        original_image, message, stego_with_cluster, threshold, eps, min_samples, False
    )
    if not success2:
        print_error(f"Encode with cluster gagal: {msg2}")
        return
    print_success("Encode dengan clustering berhasil")

    print("\n--- Step 3: Decode kedua hasil ---")
    success3, decoded_no_cluster = SteganographyEdge.decode_message(stego_no_cluster, threshold)
    success4, decoded_with_cluster = SteganographyEdgeAdaptive.decode_message(
        stego_with_cluster, threshold, eps, min_samples, False
    )

    if not success3:
        print_error(f"Decode no cluster gagal: {decoded_no_cluster}")
        return
    if not success4:
        print_error(f"Decode with cluster gagal: {decoded_with_cluster}")
        return
    print_success("Decode kedua metode berhasil")

    print("\n--- Step 4-5: Evaluasi & Perbandingan ---")
    success5, result = Evaluation.compare_methods(
        original_image, message,
        stego_no_cluster, decoded_no_cluster,
        stego_with_cluster, decoded_with_cluster
    )

    if not success5:
        print_error(f"Evaluasi gagal: {result}")
        return

    # Tampilkan hasil
    print("\n" + "="*70)
    print("HASIL PERBANDINGAN METODE")
    print("="*70)

    print("\n1. TANPA CLUSTERING (Edge-Based LSB):")
    print(f"   PSNR: {result['no_clustering']['psnr']:.2f} dB - {Evaluation.interpret_psnr(result['no_clustering']['psnr'])}")
    print(f"   BER:  {result['no_clustering']['ber_percentage']:.4f}% - {Evaluation.interpret_ber(result['no_clustering']['ber'])}")
    print(f"   Accuracy: {result['no_clustering']['accuracy']:.2f}%")

    print("\n2. DENGAN CLUSTERING (DBSCAN + Edge-Based LSB):")
    print(f"   PSNR: {result['with_clustering']['psnr']:.2f} dB - {Evaluation.interpret_psnr(result['with_clustering']['psnr'])}")
    print(f"   BER:  {result['with_clustering']['ber_percentage']:.4f}% - {Evaluation.interpret_ber(result['with_clustering']['ber'])}")
    print(f"   Accuracy: {result['with_clustering']['accuracy']:.2f}%")

    print("\n3. PERBANDINGAN:")
    print(f"   PSNR: {result['comparison']['psnr_better']} lebih baik ({abs(result['comparison']['psnr_difference']):.2f} dB)")
    print(f"   BER:  {result['comparison']['ber_better']} lebih baik ({abs(result['comparison']['ber_difference']*100):.4f}%)")

    print("\n4. KESIMPULAN:")
    if result['with_clustering']['psnr'] > result['no_clustering']['psnr']:
        print("   ✓ Clustering menghasilkan PSNR lebih tinggi (kualitas gambar lebih baik)")
    else:
        print("   ✗ Clustering menghasilkan PSNR lebih rendah")

    if result['with_clustering']['ber'] < result['no_clustering']['ber']:
        print("   ✓ Clustering menghasilkan BER lebih rendah (decode lebih akurat)")
    else:
        print("   ✗ Clustering menghasilkan BER lebih tinggi")

    print("="*70)

    # Cleanup temp files
    if os.path.exists(stego_no_cluster):
        os.remove(stego_no_cluster)
    if os.path.exists(stego_with_cluster):
        os.remove(stego_with_cluster)


def display_menu():
    """Menampilkan menu utama"""
    print("\n=== METODE STEGANOGRAPHY ===")
    print("\nA. Standard LSB Steganography:")
    print("1. Sembunyikan pesan dalam gambar (Encode)")
    print("2. Ekstrak pesan dari gambar (Decode)")
    print("3. Cek kapasitas gambar")
    print("\nB. Edge-Based Steganography (Sobel):")
    print("4. Sembunyikan pesan di edge gambar (Encode Edge)")
    print("5. Ekstrak pesan dari edge gambar (Decode Edge)")
    print("6. Visualisasi edge detection")
    print("\nC. Edge-Based + DBSCAN Clustering:")
    print("7. Visualisasi clustering edge")
    print("8. Encode dengan clustered edge")
    print("9. Decode dengan clustered edge")
    print("\nD. Evaluasi & Perbandingan:")
    print("10. Perbandingan metode (PSNR & BER)")
    print("\n11. Keluar")
    print("-"*50)


def run():
    """Fungsi utama untuk menjalankan program"""
    while True:
        print_header()
        display_menu()

        choice = input("Pilih menu (1-11): ").strip()

        if choice == '1':
            encode_menu()
        elif choice == '2':
            decode_menu()
        elif choice == '3':
            capacity_menu()
        elif choice == '4':
            encode_edge_menu()
        elif choice == '5':
            decode_edge_menu()
        elif choice == '6':
            visualize_edge_menu()
        elif choice == '7':
            cluster_visualization_menu()
        elif choice == '8':
            encode_clustered_menu()
        elif choice == '9':
            decode_clustered_menu()
        elif choice == '10':
            compare_methods_menu()
        elif choice == '11':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        else:
            print_error("Pilihan tidak valid! Silakan pilih 1-11.")

        input("\nTekan Enter untuk melanjutkan...")
