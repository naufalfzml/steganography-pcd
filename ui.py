import os
from steganography import Steganography
from stegano_edge import SteganographyEdge
from stegano_edge_clustered import SteganographyEdgeClustered
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
    threshold_input = input("Threshold edge detection (range: 0-255): ").strip()
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
    threshold_input = input("Threshold edge detection : ").strip()
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

    threshold = int(input("Threshold edge detection : ").strip() or 50)
    eps = float(input("DBSCAN eps : ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples : ").strip() or 5)
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

    threshold = int(input("Threshold edge detection : ").strip() or 50)
    eps = float(input("DBSCAN eps : ").strip() or 4)
    min_samples = int(input("DBSCAN min_samples : ").strip() or 3)
    variance_percentile = int(input("Variance Percentile (range: 0-100): ").strip() or 90)

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
        percentage = result.get('edge_percentage', 0)
        print(f"ℹ Edge pixels ({pixel_type}): {result.get('edge_pixels', 'N/A')} ({percentage:.2f}% dari total)")
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
    threshold = int(input("Threshold : ").strip() or 50)
    eps = float(input("DBSCAN eps : ").strip() or 3)
    min_samples = int(input("DBSCAN min_samples : ").strip() or 5)

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
    """Menu perbandingan 3 metode"""
    print("\n--- PERBANDINGAN: Edge vs Clustered Edge vs Adaptive Edge ---")
    print("\nProses ini akan:")
    print("1. Encode pesan dengan metode Edge-Based (standar)")
    print("2. Encode pesan dengan metode Clustered Edge (Grouped/Isolated)")
    print("3. Encode pesan dengan metode Adaptive Edge")
    print("4. Decode ketiga hasil")
    print("5. Hitung PSNR dan BER untuk perbandingan\n")

    original_image = input("Path gambar original: ").strip()
    if not os.path.exists(original_image):
        print_error(f"File tidak ditemukan!")
        return

    message = input("Pesan yang akan di-test: ")
    
    print("\nMasukkan parameter:")
    threshold = int(input("Threshold Edge (default 50): ").strip() or 50)
    eps = float(input("DBSCAN eps (default 4): ").strip() or 4)
    min_samples = int(input("DBSCAN min_samples (default 3): ").strip() or 3)
    variance_percentile = int(input("Variance Percentile (Adaptive only, default 90): ").strip() or 90)
    
    print("\nTipe pixel untuk Clustered & Adaptive (jika relevan):")
    print("1. Grouped/Clustered Pixels (Kapasitas Tinggi)")
    print("2. Isolated/Noise Pixels (Keamanan Tinggi)")
    pixel_choice = input("Pilihan (1/2, default 1): ").strip()
    use_isolated = (pixel_choice == '2')
    pixel_type_str = "Isolated" if use_isolated else "Grouped"

    # Temporary files
    stego_edge = "temp_stego_edge.png"
    stego_clustered = "temp_stego_clustered.png"
    stego_adaptive = "temp_stego_adaptive.png"

    # --- 1. Edge-Based ---
    print("\n--- 1. Metode Edge-Based ---")
    success1, msg1 = SteganographyEdge.encode_message(
        original_image, message, stego_edge, threshold
    )
    if success1: print_success("Encode Edge-Based berhasil")
    else: print_error(f"Encode Edge-Based gagal: {msg1}")

    # --- 2. Clustered Edge ---
    print(f"\n--- 2. Metode Clustered Edge ({pixel_type_str}) ---")
    success2, msg2 = SteganographyEdgeClustered.encode_message(
        original_image, message, stego_clustered, threshold, eps, min_samples, use_isolated
    )
    if success2: print_success("Encode Clustered Edge berhasil")
    else: print_error(f"Encode Clustered Edge gagal: {msg2}")

    # --- 3. Adaptive Edge ---
    print(f"\n--- 3. Metode Adaptive Edge ({pixel_type_str}) ---")
    success3, msg3 = SteganographyEdgeAdaptive.encode_message(
        original_image, message, stego_adaptive, threshold, eps, min_samples, use_isolated, variance_percentile
    )
    if success3: print_success("Encode Adaptive Edge berhasil")
    else: print_error(f"Encode Adaptive Edge gagal: {msg3}")

    # --- Evaluasi ---
    print("\n--- Evaluasi & Hasil ---")
    print(f"{'METODE':<20} | {'PSNR (dB)':<15} | {'BER (%)':<15} | {'KAPASITAS (est)':<15}")
    print("-" * 75)

    # Eval Edge
    if success1 and os.path.exists(stego_edge):
        s_dec, decoded = SteganographyEdge.decode_message(stego_edge, threshold)
        psnr = Evaluation.calculate_psnr(original_image, stego_edge)[1]
        ber = Evaluation.calculate_ber(message, decoded)[1]['ber_percentage']
        # Estimate capacity
        _, cap = SteganographyEdge.get_capacity(original_image, threshold)
        cap_val = cap.get('max_chars', 0)
        print(f"{'Edge-Based':<20} | {psnr:<15.2f} | {ber:<15.4f} | {cap_val:<15}")
    else:
        print(f"{'Edge-Based':<20} | {'Gagal':<15} | {'Gagal':<15} | {'-':<15}")

    # Eval Clustered
    if success2 and os.path.exists(stego_clustered):
        s_dec, decoded = SteganographyEdgeClustered.decode_message(stego_clustered, threshold, eps, min_samples, use_isolated)
        psnr = Evaluation.calculate_psnr(original_image, stego_clustered)[1]
        ber = Evaluation.calculate_ber(message, decoded)[1]['ber_percentage']
        _, cap = SteganographyEdgeClustered.get_capacity(original_image, threshold, eps, min_samples, use_isolated)
        cap_val = cap.get('max_chars', 0)
        print(f"{'Clustered Edge':<20} | {psnr:<15.2f} | {ber:<15.4f} | {cap_val:<15}")
    else:
        print(f"{'Clustered Edge':<20} | {'Gagal':<15} | {'Gagal':<15} | {'-':<15}")

    # Eval Adaptive
    if success3 and os.path.exists(stego_adaptive):
        s_dec, decoded = SteganographyEdgeAdaptive.decode_message(stego_adaptive, threshold, eps, min_samples, use_isolated)
        psnr = Evaluation.calculate_psnr(original_image, stego_adaptive)[1]
        ber = Evaluation.calculate_ber(message, decoded)[1]['ber_percentage']
        _, cap = SteganographyEdgeAdaptive.get_capacity(original_image, threshold, eps, min_samples, use_isolated, variance_percentile)
        cap_val = cap.get('max_chars', 0)
        print(f"{'Adaptive Edge':<20} | {psnr:<15.2f} | {ber:<15.4f} | {cap_val:<15}")
    else:
        print(f"{'Adaptive Edge':<20} | {'Gagal':<15} | {'Gagal':<15} | {'-':<15}")

    # Cleanup
    for f in [stego_edge, stego_clustered, stego_adaptive]:
        if os.path.exists(f): os.remove(f)


def robustness_test_menu():
    """Menu pengujian robustness dengan Salt & Pepper Noise"""
    print("\n--- PENGUJIAN ROBUSTNESS (Salt & Pepper Noise) ---")
    print("Pengujian ketahanan pesan terhadap noise")

    image_path = input("Path gambar original: ").strip()
    if not os.path.exists(image_path):
        print_error(f"File tidak ditemukan!")
        return

    message = input("Pesan yang akan di-test: ")
    
    # ✅ TAMBAHAN: Input noise level
    noise_input = input("Noise level (0.0-1.0, default 0.01 untuk 1%): ").strip()
    noise_level = 0.01
    if noise_input:
        try:
            noise_level = float(noise_input)
            if noise_level < 0 or noise_level > 1:
                print_error("Noise level harus antara 0.0 dan 1.0. Menggunakan default: 0.01")
                noise_level = 0.01
        except ValueError:
            print_error("Input tidak valid. Menggunakan default: 0.01")
    
    print(f"\nℹ Noise level: {noise_level*100:.1f}%")
    
    print("\nPilih Algoritma:")
    print("1. Edge-Based (Standard)")
    print("2. Clustered Edge")
    print("3. Adaptive Edge")
    
    algo_choice = input("Pilihan (1-3): ").strip()
    
    threshold = int(input("Threshold Edge (default 50): ").strip() or 50)
    
    eps = 4
    min_samples = 3
    variance_percentile = 90
    use_isolated = False
    
    if algo_choice in ['2', '3']:
        eps = float(input("DBSCAN eps (default 4): ").strip() or 4)
        min_samples = int(input("DBSCAN min_samples (default 3): ").strip() or 3)
        
        print("\nTipe pixel:")
        print("1. Grouped/Clustered Pixels")
        print("2. Isolated/Noise Pixels")
        if input("Pilihan (1/2, default 1): ").strip() == '2':
            use_isolated = True
            
    if algo_choice == '3':
        variance_percentile = int(input("Variance Percentile (default 90): ").strip() or 90)

    # Temp files
    stego_path = "temp_robust_stego.png"
    noisy_path = "temp_robust_noisy.png"
    
    try:
        # 1. Encode
        print("\n1. Encoding pesan...")
        success_enc = False
        msg_enc = ""
        
        if algo_choice == '1':
            success_enc, msg_enc = SteganographyEdge.encode_message(
                image_path, message, stego_path, threshold
            )
        elif algo_choice == '2':
            success_enc, msg_enc = SteganographyEdgeClustered.encode_message(
                image_path, message, stego_path, threshold, eps, min_samples, use_isolated
            )
        elif algo_choice == '3':
            success_enc, msg_enc = SteganographyEdgeAdaptive.encode_message(
                image_path, message, stego_path, threshold, eps, min_samples, use_isolated, variance_percentile
            )
        else:
            print_error("Pilihan algoritma tidak valid")
            return

        if not success_enc:
            print_error(f"Encode gagal: {msg_enc}")
            return
        print_success("Encode berhasil")

        # 2. Add Noise (SEKARANG LEBIH REALISTIS!)
        print(f"\n2. Menambahkan Salt & Pepper Noise ({noise_level*100:.1f}%)...")
        success_noise, msg_noise = Evaluation.add_salt_and_pepper_noise(
            stego_path, noisy_path, noise_level, random_seed=42
        )
        if not success_noise:
            print_error(f"Gagal menambah noise: {msg_noise}")
            return
        print_success(msg_noise)  # ← Tampilkan detail statistik noise

        # 3. Decode
        print("\n3. Decoding pesan dari gambar bernoise...")
        success_dec = False
        decoded_message = ""
        
        if algo_choice == '1':
            success_dec, decoded_message = SteganographyEdge.decode_message(noisy_path, threshold)
        elif algo_choice == '2':
            success_dec, decoded_message = SteganographyEdgeClustered.decode_message(
                noisy_path, threshold, eps, min_samples, use_isolated
            )
        elif algo_choice == '3':
            success_dec, decoded_message = SteganographyEdgeAdaptive.decode_message(
                noisy_path, threshold, eps, min_samples, use_isolated
            )

        # 4. Result
        print("\n" + "="*60)
        print("HASIL PENGUJIAN ROBUSTNESS")
        print("-"*60)
        print(f"Algoritma       : {['Standard Edge', 'Clustered Edge', 'Adaptive Edge'][int(algo_choice)-1]}")
        print(f"Noise Level     : {noise_level*100:.1f}%")
        print(f"Pesan Asli      : {message[:50]}{'...' if len(message) > 50 else ''}")
        
        if success_dec:
            print(f"Pesan Decode    : {decoded_message[:50]}{'...' if len(decoded_message) > 50 else ''}")
            success_ber, ber_res = Evaluation.calculate_ber(message, decoded_message)
            if success_ber:
                print(f"\nMetrik:")
                print(f"  BER           : {ber_res['ber_percentage']:.4f}%")
                print(f"  Akurasi       : {ber_res['accuracy']:.4f}%")
                print(f"  Error Bits    : {ber_res['error_bits']} / {ber_res['total_bits']}")
                print(f"  Status        : {Evaluation.interpret_ber(ber_res['ber'])}")
            else:
                print("Gagal menghitung BER")
        else:
            print(f"Pesan Decode    : [GAGAL] {decoded_message}")
            print("\n⚠ Decode gagal! Pesan tidak dapat di-extract dari gambar bernoise.")
            
        print("="*60)

    finally:
        # Cleanup
        if os.path.exists(stego_path): os.remove(stego_path)
        if os.path.exists(noisy_path): os.remove(noisy_path)


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
    print("10. Perbandingan 3 metode (Edge, Clustered, Adaptive)")
    print("11. Pengujian Robustness (Salt & Pepper 1%)")
    print("\n12. Keluar")
    print("-"*50)


def run():
    """Fungsi utama untuk menjalankan program"""
    while True:
        print_header()
        display_menu()

        choice = input("Pilih menu (1-12): ").strip()

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
            robustness_test_menu()
        elif choice == '12':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        else:
            print_error("Pilihan tidak valid! Silakan pilih 1-12.")

        input("\nTekan Enter untuk melanjutkan...")