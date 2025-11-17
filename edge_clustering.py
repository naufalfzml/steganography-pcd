import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from edge_detection import EdgeDetection


class EdgeClustering:
    """
    Kelas untuk clustering edge pixels menggunakan DBSCAN
    Tujuan: Mengelompokkan edge pixels yang berdekatan untuk optimasi steganography
    """

    @staticmethod
    def cluster_edge_pixels(image_path, threshold=50, eps=6, min_samples=20):
        """
        Clustering edge pixels menggunakan DBSCAN dengan feature tambahan (grayscale intensity)

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection (default: 50)
            eps: Maximum distance between two samples (default: 6, optimal untuk edge)
            min_samples: Minimum samples in a neighborhood (default: 20, optimal untuk edge)

        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        try:
            # Validasi parameter
            if eps <= 0 or min_samples < 2:
                return False, "Parameter DBSCAN tidak valid (eps > 0, min_samples >= 2)"
            if threshold < 0 or threshold > 255:
                return False, "Threshold harus antara 0-255"

            # Dapatkan edge pixels
            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, f"Tidak ada edge ditemukan dengan threshold {threshold}"

            # Buka gambar untuk mendapatkan grayscale value
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)

            # Tambahkan grayscale value sebagai feature ketiga
            # Format: [[x1, y1, gray1], [x2, y2, gray2], ...]
            coordinates_with_intensity = []
            for x, y in edge_coords:
                gray_value = img_array[y, x]
                coordinates_with_intensity.append([x, y, gray_value])

            coordinates_array = np.array(coordinates_with_intensity)

            # Normalisasi features agar spatial dan intensity seimbang
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates_array)

            # Set random seed untuk konsistensi
            np.random.seed(42)

            # Terapkan DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(coordinates_scaled)

            # Analisis hasil clustering
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Pisahkan edge pixels yang ter-cluster vs noise
            clustered_coords = []
            noise_coords = []

            for i, label in enumerate(labels):
                coord = edge_coords[i]
                if label == -1:
                    noise_coords.append(coord)
                else:
                    clustered_coords.append(coord)

            # CRITICAL: Sort untuk konsistensi urutan
            clustered_coords.sort()
            noise_coords.sort()

            # Hitung statistik per cluster
            cluster_stats = {}
            for label in set(labels):
                if label == -1:  # Skip noise
                    continue
                cluster_size = list(labels).count(label)
                cluster_stats[label] = cluster_size

            result = {
                'total_edge_pixels': len(edge_coords),
                'n_clusters': n_clusters,
                'clustered_pixels': len(clustered_coords),
                'noise_pixels': n_noise,
                'clustered_coords': clustered_coords,
                'noise_coords': noise_coords,
                'all_coords': edge_coords,
                'labels': labels,
                'cluster_stats': cluster_stats,
                'eps': eps,
                'min_samples': min_samples,
                'threshold': threshold
            }

            return True, result

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_optimized_edge_pixels(image_path, threshold=50, eps=6, min_samples=20, use_isolated=False):
        """
        Mendapatkan edge pixels yang sudah dioptimasi dengan clustering

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            use_isolated: Jika True, hanya gunakan isolated edge pixels (noise, lebih aman)
                         Jika False, hanya gunakan grouped edge pixels (clustered, kapasitas lebih besar)

        Returns:
            tuple: (success: bool, result: list of (x,y) atau error message)
        """
        success, cluster_result = EdgeClustering.cluster_edge_pixels(
            image_path, threshold, eps, min_samples
        )

        if not success:
            return False, cluster_result

        if use_isolated:
            # Gunakan noise pixels (isolated edges)
            # Lebih aman karena edge yang terisolasi lebih sulit dideteksi
            coords = cluster_result['noise_coords']
        else:
            # Gunakan clustered pixels (grouped edges)
            # Lebih banyak kapasitas
            coords = cluster_result['clustered_coords']

        if len(coords) == 0:
            msg = "isolated" if use_isolated else "grouped"
            return False, f"Tidak ada {msg} edge pixels ditemukan"

        return True, coords

    @staticmethod
    def visualize_clusters(image_path, output_path, threshold=50, eps=6, min_samples=20):
        """
        Visualisasi hasil clustering dengan warna berbeda per cluster

        Args:
            image_path: Path ke gambar
            output_path: Path untuk save visualisasi
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples

        Returns:
            tuple: (success: bool, message: str dengan statistik)
        """
        try:
            # Clustering
            success, result = EdgeClustering.cluster_edge_pixels(
                image_path, threshold, eps, min_samples
            )

            if not success:
                return False, result

            # Buka gambar untuk dimensi
            img = Image.open(image_path)
            width, height = img.size

            # Buat canvas RGB untuk visualisasi warna
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Generate warna untuk setiap cluster
            n_clusters = result['n_clusters']
            np.random.seed(42)  # Konsistensi warna
            colors = np.random.randint(50, 255, size=(max(n_clusters, 1), 3))

            # Gambar edge pixels dengan warna cluster
            labels = result['labels']
            all_coords = result['all_coords']

            for i, (x, y) in enumerate(all_coords):
                label = labels[i]
                if label == -1:
                    # Noise = merah
                    canvas[y, x] = [255, 0, 0]
                else:
                    # Cluster = warna random
                    canvas[y, x] = colors[label % len(colors)]

            # Save
            img_result = Image.fromarray(canvas)
            img_result.save(output_path)

            # Return dengan statistik lengkap
            stats_msg = (
                f"Visualisasi berhasil disimpan di: {output_path}\n"
                f"Total edge pixels: {result['total_edge_pixels']}\n"
                f"Jumlah clusters: {result['n_clusters']}\n"
                f"Clustered pixels: {result['clustered_pixels']}\n"
                f"Isolated pixels (noise): {result['noise_pixels']}"
            )

            return True, stats_msg

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_capacity_clustered(image_path, threshold=50, eps=6, min_samples=20, use_isolated=False):
        """
        Hitung kapasitas steganography dengan clustering

        Args:
            image_path: Path ke gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples
            use_isolated: True = gunakan isolated pixels, False = gunakan grouped pixels

        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        success, coords = EdgeClustering.get_optimized_edge_pixels(
            image_path, threshold, eps, min_samples, use_isolated
        )

        if not success:
            return False, coords

        # Hitung kapasitas
        edge_count = len(coords)
        max_bits = edge_count * 3  # RGB = 3 bit per pixel
        
        # Gunakan length prefix (16 bits) instead of delimiter
        max_bits_available = max_bits - 16  # Reserve 16 bits for length
        max_chars = max_bits_available // 8

        # Get total pixels
        img = Image.open(image_path)
        width, height = img.size
        total_pixels = width * height

        edge_percentage = (edge_count / total_pixels) * 100

        result = {
            'edge_pixels': edge_count,
            'edge_percentage': edge_percentage,
            'max_chars': max_chars,
            'threshold': threshold,
            'eps': eps,
            'min_samples': min_samples,
            'use_isolated': use_isolated,
            'pixel_type': 'isolated' if use_isolated else 'grouped'
        }

        return True, result

    @staticmethod
    def get_cluster_info(image_path, threshold=50, eps=6, min_samples=20):
        """
        Mendapatkan informasi detail tentang clustering

        Args:
            image_path: Path ke gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples

        Returns:
            tuple: (success: bool, info: dict)
        """
        success, result = EdgeClustering.cluster_edge_pixels(
            image_path, threshold, eps, min_samples
        )

        if not success:
            return False, result

        info = {
            'total_edge_pixels': result['total_edge_pixels'],
            'n_clusters': result['n_clusters'],
            'clustered_pixels': result['clustered_pixels'],
            'isolated_pixels': result['noise_pixels'],
            'cluster_stats': result['cluster_stats'],
            'largest_cluster': max(result['cluster_stats'].values()) if result['cluster_stats'] else 0,
            'smallest_cluster': min(result['cluster_stats'].values()) if result['cluster_stats'] else 0,
            'avg_cluster_size': np.mean(list(result['cluster_stats'].values())) if result['cluster_stats'] else 0
        }

        return True, info