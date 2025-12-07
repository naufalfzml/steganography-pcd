from PIL import Image
import numpy as np
from edge_clustering import EdgeClustering

class SteganographyEdgeAdaptive:
    # Steganography Adaptive dengan Block-Based Flagging
    # Flag '1' = High Variance (2-bit), Flag '0' = Low Variance (1-bit)
    
    CHUNK_SIZE = 8

    @staticmethod
    def calculate_chunk_variance(image, coords_chunk):
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
            img = Image.open(image_path).convert('RGB')
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            
            if not success: return False, edge_coords
            
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            total_pixels = len(edge_coords)
            if total_pixels == 0:
                 return True, {'max_chars': 0, 'edge_pixels': 0}

            chunk_variances = []
            
            num_chunks = len(edge_coords) // SteganographyEdgeAdaptive.CHUNK_SIZE
            
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                if len(chunk) < SteganographyEdgeAdaptive.CHUNK_SIZE:
                    continue
                    
                v = SteganographyEdgeAdaptive.calculate_chunk_variance(img, chunk)
                chunk_variances.append(v)

            if not chunk_variances:
                return False, "Tidak ada chunk valid yang terbentuk"

            global_threshold = np.percentile(chunk_variances, variance_percentile)

            total_bits_capacity = 0
            high_chunks = 0
            low_chunks = 0
            
            for v in chunk_variances:
                available_channels = (SteganographyEdgeAdaptive.CHUNK_SIZE * 3) - 1
                
                if v >= global_threshold:
                    bits = available_channels * 2 
                    high_chunks += 1
                else:
                    bits = available_channels * 1
                    low_chunks += 1
                
                total_bits_capacity += bits

            max_bits_data = total_bits_capacity - 32
            max_chars = max_bits_data // 8
            
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
                       variance_percentile=70):
        try:
            img = Image.open(image_path).convert('RGB')
            pixels = img.load()
            
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            if not success: return False, "Gagal deteksi edge"
            
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            chunk_variances = []
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                v = SteganographyEdgeAdaptive.calculate_chunk_variance(img, chunk)
                chunk_variances.append(v)
            
            if not chunk_variances: return False, "Tidak ada edge"
            global_threshold = np.percentile(chunk_variances, variance_percentile)
            
            msg_len = len(message)
            header_bits = format(msg_len, '032b') 
            data_bits = ''.join([format(ord(c), '08b') for c in message])
            full_bits = header_bits + data_bits
            
            bit_idx = 0
            total_bits = len(full_bits)
            
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                if bit_idx >= total_bits: break
                
                chunk_coords = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                chunk_variance = chunk_variances[i // SteganographyEdgeAdaptive.CHUNK_SIZE]
                
                is_high = chunk_variance >= global_threshold
                
                chunk_channels = []
                for x, y in chunk_coords:
                    chunk_channels.extend([(x, y, 0), (x, y, 1), (x, y, 2)])
                
                fx, fy, fc = chunk_channels[0]
                flag_val = 1 if is_high else 0
                
                p_list = list(pixels[fx, fy])
                p_list[fc] = (p_list[fc] & ~1) | flag_val
                pixels[fx, fy] = tuple(p_list)
                
                for cx, cy, cc in chunk_channels[1:]:
                    if bit_idx >= total_bits: break
                    
                    p_list = list(pixels[cx, cy])
                    
                    if is_high:
                        bits_needed = 2
                        bits_avail = total_bits - bit_idx
                        to_embed = min(bits_needed, bits_avail)
                        
                        chunk_msg = full_bits[bit_idx : bit_idx + to_embed]
                        val = int(chunk_msg, 2)
                        
                        if to_embed == 2:
                            p_list[cc] = (p_list[cc] & ~3) | val
                        else:
                            p_list[cc] = (p_list[cc] & ~1) | val
                            
                        bit_idx += to_embed
                    else:
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
        try:
            img = Image.open(image_path).convert('RGB')
            pixels = img.load()
            
            success, edge_coords = EdgeClustering.get_optimized_edge_pixels(
                image_path, threshold, eps, min_samples, use_isolated
            )
            if not success: 
                return False, "Gagal deteksi edge"
            
            edge_coords.sort(key=lambda p: (p[1], p[0]))
            
            decoded_bits = ""
            
            if expected_length is not None:
                target_len = 32 + (expected_length * 8)
                reading_header = False
            else:
                target_len = 32
                reading_header = True
            
            for i in range(0, len(edge_coords), SteganographyEdgeAdaptive.CHUNK_SIZE):
                chunk_coords = edge_coords[i:i + SteganographyEdgeAdaptive.CHUNK_SIZE]
                
                chunk_channels = []
                for x, y in chunk_coords:
                    chunk_channels.extend([(x, y, 0), (x, y, 1), (x, y, 2)])
                
                fx, fy, fc = chunk_channels[0]
                flag = pixels[fx, fy][fc] & 1
                is_high = (flag == 1)
                
                for cx, cy, cc in chunk_channels[1:]:
                    if expected_length is None and not reading_header and len(decoded_bits) >= target_len:
                        break
                    if expected_length is not None and len(decoded_bits) >= target_len:
                        break
                        
                    val = pixels[cx, cy][cc]
                    
                    if is_high:
                        bits = format(val & 3, '02b')
                        decoded_bits += bits
                    else:
                        bits = str(val & 1)
                        decoded_bits += bits
                        
                    if expected_length is None and reading_header and len(decoded_bits) >= 32:
                        len_bits = decoded_bits[:32]
                        msg_len = int(len_bits, 2)
                        
                        decoded_bits = decoded_bits[32:]
                        target_len = msg_len * 8
                        reading_header = False
                        
                if expected_length is None and not reading_header and len(decoded_bits) >= target_len:
                    break
                if expected_length is not None and len(decoded_bits) >= target_len:
                    break
            
            if expected_length is not None:
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