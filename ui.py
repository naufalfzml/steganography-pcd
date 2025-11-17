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
    """Menu encode dengan clustering (Adaptive)"""
    print("\n--- ENCODE (ADAPTIVE EDGE): Sembunyikan Pesan di Edge ---")

    image_path = input("Path gambar input: ").strip()
    if not os.path.exists(image_path):
        print_error(f"File '{image_path}' tidak ditemukan!")
        return

    threshold = int(input("Threshold edge detection (default: 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default: 4): ").strip() or 4)
    min_samples = int(input("DBSCAN min_samples (default: 3): ").strip() or 3)
    variance_percentile = int(input("Variance Percentile (default: 90, range: 0-100): ").strip() or 90)

    print("\n  Pilih tipe edge pixels:")
    print("  1. Clustered pixels (grouped edges, kapasitas >)")
    print("  2. Isolated pixels (noise edges, keamanan >)")
    choice = input("  Pilihan (1/2): ").strip()
    use_isolated = (choice == '2')

    print("\nMenganalisis edge dan clustering...")
    success, result = SteganographyEdgeAdaptive.get_capacity(
        image_path, threshold, eps, min_samples, use_isolated, variance_percentile
    )

    if success:
        pixel_type = "isolated" if use_isolated else "grouped"
        print(f"ℹ Edge pixels ({pixel_type}): {result.get('edge_pixels', 'N/A')}")
        print(f"ℹ Kapasitas maksimal: {result.get('max_chars', 'N/A')} karakter (mode: {result.get('mode', 'N/A')})")
    else:
        print_error(result)
        return

    message = input("\nMasukkan pesan: ")

    if len(message) > result.get('max_chars', 0):
        print_error(f"Pesan terlalu panjang! Maksimal {result.get('max_chars', 0)} karakter.")
        return

    output_path = input("Path output: ").strip()

    print("\nMemproses...")
    success, msg = SteganographyEdgeAdaptive.encode_message(
        image_path, message, output_path, threshold, eps, min_samples, use_isolated, variance_percentile
    )

    if success:
        print_success(msg)
        print(f"ℹ Parameter: th={threshold}, eps={eps}, ms={min_samples}, iso={use_isolated}, var_p={variance_percentile}")
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
    print("\n--- PERBANDINGAN: Edge vs Adaptive Edge ---")
    print("\nProses ini akan:")
    print("1. Encode pesan dengan metode Edge-Based (standar)")
    print("2. Encode pesan dengan metode Adaptive Edge (clustering + adaptive bit)")
    print("3. Decode kedua hasil")
    print("4. Hitung PSNR dan BER untuk perbandingan\n")

    original_image = input("Path gambar original: ").strip()
    if not os.path.exists(original_image):
        print_error(f"File tidak ditemukan!")
        return

    message = input("Pesan yang akan di-test: ")
    
    print("\nMasukkan parameter untuk kedua metode:")
    threshold = int(input("Threshold (default: 50): ").strip() or 50)
    
    print("\nMasukkan parameter untuk metode Adaptive (jika tidak diisi, akan diskip):")
    eps_input = input("DBSCAN eps (default: 4): ").strip()
    min_samples_input = input("DBSCAN min_samples (default: 3): ").strip()
    variance_percentile_input = input("Variance Percentile (default: 90): ").strip()

    run_adaptive = eps_input and min_samples_input and variance_percentile_input

    eps = float(eps_input or 4)
    min_samples = int(min_samples_input or 3)
    variance_percentile = int(variance_percentile_input or 90)

    # Temporary files
    stego_edge = "temp_stego_edge.png"
    stego_adaptive = "temp_stego_adaptive.png"

    print("\n--- Step 1: Encode metode Edge-Based ---")
    success1, msg1 = SteganographyEdge.encode_message(
        original_image, message, stego_edge, threshold
    )
    if not success1:
        print_error(f"Encode Edge-Based gagal: {msg1}")
        return
    print_success("Encode Edge-Based berhasil")

    decoded_edge, decoded_adaptive = "", ""
    if run_adaptive:
        print("\n--- Step 2: Encode metode Adaptive Edge ---")
        # Menggunakan clustered pixels (use_isolated=False) untuk perbandingan
        success2, msg2 = SteganographyEdgeAdaptive.encode_message(
            original_image, message, stego_adaptive, threshold, eps, min_samples, False, variance_percentile
        )
        if not success2:
            print_error(f"Encode Adaptive Edge gagal: {msg2}")
        else:
            print_success("Encode Adaptive Edge berhasil")

    print("\n--- Step 3: Decode kedua hasil ---")
    success3, decoded_edge = SteganographyEdge.decode_message(stego_edge, threshold)
    if not success3:
        print_error(f"Decode Edge-Based gagal: {decoded_edge}")
        return
    print_success("Decode Edge-Based berhasil.")

    if run_adaptive and os.path.exists(stego_adaptive):
        success4, decoded_adaptive = SteganographyEdgeAdaptive.decode_message(
            stego_adaptive, threshold, eps, min_samples, False
        )
        if not success4:
            print_error(f"Decode Adaptive Edge gagal: {decoded_adaptive}")
        else:
            print_success("Decode Adaptive Edge berhasil.")

    print("\n--- Step 4-5: Evaluasi & Perbandingan ---")
    
    # Evaluasi untuk Edge-Based
    psnr_edge_s, psnr_edge = Evaluation.calculate_psnr(original_image, stego_edge)
    ber_edge_s, ber_edge_res = Evaluation.calculate_ber(message, decoded_edge)

    print("\n" + "="*70)
    print("HASIL PERBANDINGAN METODE")
    print("="*70)

    if not (psnr_edge_s and ber_edge_s):
        print_error("Gagal mengevaluasi metode Edge-Based.")
    else:
        print("\n1. EDGE-BASED (Metode Dasar):")
        print(f"   PSNR: {psnr_edge:.2f} dB - {Evaluation.interpret_psnr(psnr_edge)}")
        print(f"   BER:  {ber_edge_res['ber_percentage']:.4f}% - {Evaluation.interpret_ber(ber_edge_res['ber'])}")

    # Evaluasi untuk Adaptive
    if run_adaptive and os.path.exists(stego_adaptive):
        psnr_adaptive_s, psnr_adaptive = Evaluation.calculate_psnr(original_image, stego_adaptive)
        ber_adaptive_s, ber_adaptive_res = Evaluation.calculate_ber(message, decoded_adaptive)
        
        if not (psnr_adaptive_s and ber_adaptive_s):
            print_error("Gagal mengevaluasi metode Adaptive Edge.")
        else:
            print("\n2. ADAPTIVE EDGE (Clustering + Adaptive Bit):")
            print(f"   PSNR: {psnr_adaptive:.2f} dB - {Evaluation.interpret_psnr(psnr_adaptive)}")
            print(f"   BER:  {ber_adaptive_res['ber_percentage']:.4f}% - {Evaluation.interpret_ber(ber_adaptive_res['ber'])}")

            print("\n3. PERBANDINGAN:")
            psnr_diff = psnr_adaptive - psnr_edge
            ber_diff = ber_adaptive_res['ber'] - ber_edge_res['ber']
            
            psnr_better = "Adaptive" if psnr_diff > 0 else "Edge-Based"
            ber_better = "Adaptive" if ber_diff < 0 else "Edge-Based"

            print(f"   PSNR: {psnr_better} lebih baik ({abs(psnr_diff):.2f} dB)")
            print(f"   BER:  {ber_better} lebih baik ({abs(ber_diff*100):.4f}%)")
    
    print("="*70)

    # Cleanup temp files
    if os.path.exists(stego_edge):
        os.remove(stego_edge)
    if os.path.exists(stego_adaptive):
        os.remove(stego_adaptive)


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