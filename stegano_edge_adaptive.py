from PIL import Image
import numpy as np
from edge_clustering import EdgeClustering

class SteganographyEdgeAdaptive:
    """
    Steganography Adaptive dengan Sistem FLAGGING (Block-Based)
    
    Konsep:
    - Pixel edge diproses dalam grup (CHUNK_SIZE = 8 pixel).
    - Setiap chunk diawali 1 bit FLAG di channel pertama pixel pertama.
    - Flag '1' = High Variance (Mode 2-bit untuk sisa chunk).
    - Flag '0' = Low Variance (Mode 1-bit untuk sisa chunk).
    - Decoder membaca Flag dulu, baru tahu cara baca sisanya.
    - HASIL: 100% Sinkron, Kapasitas tetap Adaptif.
    """
    
    CHUNK_SIZE = 8  # Jumlah pixel per grup analisis

    @staticmethod
    def calculate_chunk_variance(image, coords_chunk):
        """Hitung rata-rata variance untuk satu grup pixel"""
        variances = []
        gray = image.convert('L')
        arr = np.array(gray)
        h, w = arr.shape
        
        for x, y in coords_chunk:
            y_start, y_end = max(0, y-1), min(h, y+2)
            x_start, x_end = max(0, x-1), min(w, x+2)
            neighborhood = arr[y_start:y_end, x_start:x_end]
            variances.append(np.var(neighborhood))
            
        return np.mean(variances) if variances else 0

    @staticmethod
    def get_capacity(image_path, threshold=60, eps=0.2, min_samples=3, 
                     use_isolated=False, variance_percentile=90):
        try:
            # 1. Load Gambar & Edge (Sama seperti encode)
            img = Image.open(image_path).convert('RGB')
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            
            if not success: return False, edge_coords
            
            # Wajib sort agar urutan chunk sama dengan saat encode
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            total_pixels = len(edge_coords)
            if total_pixels == 0:
                 return True, {'max_chars': 0, 'edge_pixels': 0}

            # 2. Hitung Variance Per Chunk (Simulasi Logika Encode)
            chunk_variances = []
            
            # Kita butuh loop dulu untuk kumpulkan variance agar bisa hitung threshold
            # Karena threshold ditentukan dari percentile SELURUH chunk
            num_chunks = len(edge_coords) // SteganographyEdgeAdaptive.CHUNK_SIZE
            
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                # Skip jika chunk tidak full (sisa di akhir)
                if len(chunk) < SteganographyEdgeAdaptive.CHUNK_SIZE:
                    continue
                    
                v = SteganographyEdgeAdaptive.calculate_chunk_variance(img, chunk)
                chunk_variances.append(v)

            if not chunk_variances:
                return False, "Tidak ada chunk valid yang terbentuk"

            # 3. Hitung Global Threshold
            global_threshold = np.percentile(chunk_variances, variance_percentile)

            # 4. Hitung Total Bits Berdasarkan Klasifikasi High/Low
            total_bits_capacity = 0
            high_chunks = 0
            low_chunks = 0
            
            # Hitung per chunk
            for v in chunk_variances:
                # Dalam 1 chunk (8 pixel) terdapat 24 channel (8x3)
                # Channel pertama dipakai untuk FLAG (bukan data)
                # Sisa channel untuk data = 23 channel
                
                available_channels = (SteganographyEdgeAdaptive.CHUNK_SIZE * 3) - 1
                
                if v >= global_threshold:
                    # Mode High: 2 bit per channel
                    bits = available_channels * 2 
                    high_chunks += 1
                else:
                    # Mode Low: 1 bit per channel
                    bits = available_channels * 1
                    low_chunks += 1
                
                total_bits_capacity += bits

            # Reserve untuk header (32 bit)
            max_bits_data = total_bits_capacity - 32
            max_chars = max_bits_data // 8
            
            # Hitung persentase edge
            w, h = img.size
            edge_percentage = (total_pixels / (w * h)) * 100

            return True, {
                'edge_pixels': total_pixels,
                'edge_percentage': f"{edge_percentage:.2f}%",
                'max_bits_raw': total_bits_capacity,
                'max_chars': max_chars,
                'high_variance_chunks': high_chunks,
                'low_variance_chunks': low_chunks,
                'threshold_val': global_threshold,
                'mode': 'Real Adaptive Calculation'
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error calculating real capacity: {str(e)}"

    @staticmethod
    def encode_message(image_path, message, output_path, threshold=30,
                       eps=4, min_samples=3, use_isolated=False, 
                       variance_percentile=70): # Percentile default diturunkan biar lebih banyak yg High
        try:
            img = Image.open(image_path).convert('RGB')
            pixels = img.load()
            
            # 1. Ambil Koordinat Edge
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            if not success: return False, "Gagal deteksi edge"
            
            # Sort koordinat (Wajib konsisten)
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            # 2. Hitung Threshold Global
            # Kita sampling variance per chunk
            chunk_variances = []
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                v = SteganographyEdgeAdaptive.calculate_chunk_variance(img, chunk)
                chunk_variances.append(v)
            
            if not chunk_variances: return False, "Tidak ada edge"
            global_threshold = np.percentile(chunk_variances, variance_percentile)
            
            # 3. Siapkan Data Bitstream
            msg_len = len(message)
            # Header: 32 bit (16 bit panjang pesan + 16 bit cadangan/threshold info kalau mau)
            header_bits = format(msg_len, '032b') 
            data_bits = ''.join([format(ord(c), '08b') for c in message])
            full_bits = header_bits + data_bits
            
            bit_idx = 0
            total_bits = len(full_bits)
            
            # 4. Proses Per Chunk
            # Kita iterasi per grup pixel (Chunk)
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                if bit_idx >= total_bits: break
                
                chunk_coords = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                chunk_variance = chunk_variances[i // SteganographyEdgeAdaptive.CHUNK_SIZE]
                
                # Tentukan Mode: High (2-bit) atau Low (1-bit)
                is_high = chunk_variance >= global_threshold
                
                # --- ITERASI CHANNEL DALAM CHUNK ---
                # Kita ratakan chunk ini jadi list channel: [(x,y,R), (x,y,G), (x,y,B), ...]
                chunk_channels = []
                for x, y in chunk_coords:
                    chunk_channels.extend([(x, y, 0), (x, y, 1), (x, y, 2)])
                
                # Channel pertama di Chunk ini dikorbankan untuk FLAG
                # Flag '1' = High Mode, Flag '0' = Low Mode
                fx, fy, fc = chunk_channels[0]
                flag_val = 1 if is_high else 0
                
                p_list = list(pixels[fx, fy])
                p_list[fc] = (p_list[fc] & ~1) | flag_val # Embed Flag di LSB
                pixels[fx, fy] = tuple(p_list)
                
                # Sisa channel di chunk ini dipakai untuk data
                for cx, cy, cc in chunk_channels[1:]:
                    if bit_idx >= total_bits: break
                    
                    p_list = list(pixels[cx, cy])
                    
                    if is_high:
                        # Mode 2-bit
                        bits_needed = 2
                        bits_avail = total_bits - bit_idx
                        to_embed = min(bits_needed, bits_avail)
                        
                        chunk_msg = full_bits[bit_idx : bit_idx + to_embed]
                        val = int(chunk_msg, 2)
                        
                        if to_embed == 2:
                            p_list[cc] = (p_list[cc] & ~3) | val
                        else: # Sisa 1 bit terakhir
                            p_list[cc] = (p_list[cc] & ~1) | val
                            
                        bit_idx += to_embed
                    else:
                        # Mode 1-bit
                        bit_val = int(full_bits[bit_idx])
                        p_list[cc] = (p_list[cc] & ~1) | bit_val
                        bit_idx += 1
                        
                    pixels[cx, cy] = tuple(p_list)
            
            if bit_idx < total_bits:
                return False, f"Kapasitas kurang. {bit_idx}/{total_bits} bits ter-embed."
                
            img.save(output_path)
            return True, f"Berhasil! Mode Adaptif Flagging. Output: {output_path}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    @staticmethod
    def decode_message(image_path, threshold=30, eps=4, min_samples=3, use_isolated=False, expected_length=None):
        """
        Decode pesan dari Adaptive Edge

        Args:
            image_path: Path gambar dengan pesan
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples
            use_isolated: Tipe edge pixels
            expected_length: (Optional) Panjang pesan yang diharapkan (jumlah karakter).
                            Untuk robustness test.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            img = Image.open(image_path).convert('RGB')
            pixels = img.load()
            
            # 1. Ambil Koordinat Edge (HARUS SAMA PERSIS)
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            if not success: 
                return False, "Gagal deteksi edge"
            
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            decoded_bits = ""
            
            # ✅ SETUP MODE
            if expected_length is not None:
                # Mode Testing: Baca 32 bit header + (expected_length * 8) data
                target_len = 32 + (expected_length * 8)
                reading_header = False
            else:
                # Mode Normal: Baca header dulu
                target_len = 32
                reading_header = True
            
            # 2. Proses Decode Per Chunk
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk_coords = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                
                chunk_channels = []
                for x, y in chunk_coords:
                    chunk_channels.extend([(x, y, 0), (x, y, 1), (x, y, 2)])
                
                # BACA FLAG (Channel pertama)
                fx, fy, fc = chunk_channels[0]
                flag = pixels[fx, fy][fc] & 1
                is_high = (flag == 1)
                
                # BACA DATA (Sisa channel)
                for cx, cy, cc in chunk_channels[1:]:
                    # Cek apakah sudah selesai
                    if expected_length is None and not reading_header and len(decoded_bits) >= target_len:
                        break
                    if expected_length is not None and len(decoded_bits) >= target_len:
                        break
                        
                    val = pixels[cx, cy][cc]
                    
                    if is_high:
                        # Ambil 2 bit
                        bits = format(val & 3, '02b')
                        decoded_bits += bits
                    else:
                        # Ambil 1 bit
                        bits = str(val & 1)
                        decoded_bits += bits
                        
                    # Cek transisi Header -> Data (HANYA NORMAL MODE)
                    if expected_length is None and reading_header and len(decoded_bits) >= 32:
                        # Header selesai, parsing panjang pesan
                        len_bits = decoded_bits[:32]
                        msg_len = int(len_bits, 2)
                        
                        # Reset untuk baca body
                        decoded_bits = decoded_bits[32:]
                        target_len = msg_len * 8
                        reading_header = False
                        
                if expected_length is None and not reading_header and len(decoded_bits) >= target_len:
                    break
                if expected_length is not None and len(decoded_bits) >= target_len:
                    break
            
            # Finalisasi
            if expected_length is not None:
                # Potong header 32 bit, ambil sesuai expected length
                final_bits = decoded_bits[32 : 32 + (expected_length * 8)]
            else:
                final_bits = decoded_bits[:target_len]
                
            message = ""
            for k in range(0, len(final_bits), 8):
                byte = final_bits[k:k+8]
                if len(byte) == 8:
                    message += chr(int(byte, 2))
                    
            return True, message

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)