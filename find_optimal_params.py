
import os
from stegano_edge_adaptive import SteganographyEdgeAdaptive
from evaluation import Evaluation

def find_optimal_parameters():
    """
    Script untuk mencari parameter optimal untuk SteganographyEdgeAdaptive
    dengan target BER < 10%.
    """
    # --- Konfigurasi Pengujian ---
    IMAGE_PATH = "images/pepper.tiff"
    OUTPUT_PATH = "temp_stego_adaptive_test.png"
    MESSAGE = "a" * 256
    BER_TARGET = 10.0  # dalam persen

    # --- Parameter yang akan diuji ---
    # Berdasarkan diskusi, kita persempit rentang yang paling mungkin berhasil
    thresholds = [30, 40, 50, 60]
    eps_values = [4, 6, 8]
    min_samples_values = [3, 4, 5, 10]
    use_isolated_options = [False]  # False = grouped, True = isolated
    variance_percentiles = [50, 75, 90]

    print("--- Memulai Pencarian Parameter Optimal untuk Stegano Edge Adaptive ---")
    print(f"Gambar: {IMAGE_PATH}")
    print(f"Pesan: {len(MESSAGE)} karakter")
    print(f"Target BER: < {BER_TARGET}%")

    successful_params = []
    total_combinations = len(thresholds) * len(eps_values) * len(min_samples_values) * len(use_isolated_options) * len(variance_percentiles)
    current_combination = 0

    # --- Loop Pengujian ---
    for threshold in thresholds:
        for eps in eps_values:
            for min_samples in min_samples_values:
                for use_isolated in use_isolated_options:
                    for variance_percentile in variance_percentiles:
                        
                        current_combination += 1
                        params_str = f"th={threshold}, eps={eps}, ms={min_samples}, iso={use_isolated}, var_p={variance_percentile}"
                        print(f"[{current_combination}/{total_combinations}] Menguji: {params_str}...")

                        # 1. Encode Pesan
                        encode_success, encode_msg = SteganographyEdgeAdaptive.encode_message(
                            image_path=IMAGE_PATH,
                            message=MESSAGE,
                            output_path=OUTPUT_PATH,
                            threshold=threshold,
                            eps=eps,
                            min_samples=min_samples,
                            use_isolated=use_isolated,
                            variance_percentile=variance_percentile
                        )

                        if not encode_success:
                            # Seringkali gagal karena kapasitas tidak cukup
                            if "Pesan terlalu panjang" in encode_msg or "Tidak ada edge pixels" in encode_msg:
                                print(" -> Gagal Encode: Kapasitas tidak cukup.\n")
                            else:
                                print(f" -> Gagal Encode: {encode_msg}\n")
                            continue

                        # 2. Decode Pesan
                        decode_success, decoded_message = SteganographyEdgeAdaptive.decode_message(
                            image_path=OUTPUT_PATH,
                            threshold=threshold,
                            eps=eps,
                            min_samples=min_samples,
                            use_isolated=use_isolated
                        )

                        if not decode_success:
                            print(f" -> Gagal Decode: {decoded_message}\n")
                            continue
                        
                        # 3. Hitung BER
                        ber_success, ber_result = Evaluation.calculate_ber(MESSAGE, decoded_message)
                        if not ber_success:
                            print(f" -> Gagal Kalkulasi BER: {ber_result}\n")
                            continue
                        
                        ber_percentage = ber_result['ber_percentage']

                        # 4. Hitung PSNR
                        psnr_success, psnr_value = Evaluation.calculate_psnr(IMAGE_PATH, OUTPUT_PATH)
                        if not psnr_success:
                            psnr_value = "N/A"

                        print(f" -> Hasil: BER = {ber_percentage:.2f}%, PSNR = {psnr_value if isinstance(psnr_value, str) else f'{psnr_value:.2f} dB'}")

                        # 5. Cek apakah memenuhi target
                        if ber_percentage < BER_TARGET:
                            print("    *** SUKSES: Kombinasi ini memenuhi target! ***\n")
                            result = {
                                'threshold': threshold,
                                'eps': eps,
                                'min_samples': min_samples,
                                'use_isolated': use_isolated,
                                'variance_percentile': variance_percentile,
                                'ber': ber_percentage,
                                'psnr': psnr_value
                            }
                            successful_params.append(result)
                        else:
                            print("\n")


    # --- Ringkasan Hasil ---
    print("\n--- Pencarian Selesai ---")
    if not successful_params:
        print("Tidak ada kombinasi parameter yang ditemukan memenuhi target BER < 10%.")
        print("Coba rentang parameter yang berbeda atau periksa kembali implementasi.")
    else:
        print(f"Ditemukan {len(successful_params)} kombinasi parameter yang optimal (BER < {BER_TARGET}%):")
        # Urutkan berdasarkan BER terendah, lalu PSNR tertinggi
        sorted_params = sorted(successful_params, key=lambda p: (p['ber'], -p['psnr'] if isinstance(p['psnr'], (int, float)) else 0))
        
        for i, params in enumerate(sorted_params):
            print(f"  {i+1}. BER: {params['ber']:.2f}%, PSNR: {params['psnr'] if isinstance(params['psnr'], str) else f'{params['psnr']:.2f} dB'}")
            print(f"     Params: th={params['threshold']}, eps={params['eps']}, ms={params['min_samples']}, iso={params['use_isolated']}, var_p={params['variance_percentile']}")

    # Hapus file sementara
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    print("\nFile sementara telah dihapus.")


if __name__ == "__main__":
    find_optimal_parameters()
