import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from edge_detection import EdgeDetection


class EdgeClustering:
    """
    Kelas untuk clustering edge pixels menggunakan DBSCAN
    Tujuan: Mengelompokkan edge pixels yang berdekatan untuk optimasi steganography
    """

    @staticmethod
    def cluster_edge_pixels(image_path, threshold=50, eps=3, min_samples=5):
        """
        Clustering edge pixels menggunakan DBSCAN

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection (default: 50)
            eps: Maximum distance between two samples (default: 3)
            min_samples: Minimum samples in a neighborhood (default: 5)

        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        try:
            # Dapatkan edge pixels
            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            if len(edge_coords) == 0:
                return False, f"Tidak ada edge ditemukan dengan threshold {threshold}"

            # Convert list koordinat ke array untuk DBSCAN
            # Format: [[x1, y1], [x2, y2], ...]
            coordinates_array = np.array(edge_coords)

            # Terapkan DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(coordinates_array)

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
                'min_samples': min_samples
            }

            return True, result

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_optimized_edge_pixels(image_path, threshold=50, eps=3, min_samples=5, use_noise=False):
        """
        Mendapatkan edge pixels yang sudah dioptimasi dengan clustering

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            use_noise: Jika True, hanya gunakan noise pixels (edge isolated)
                       Jika False, hanya gunakan clustered pixels (edge grouped)

        Returns:
            tuple: (success: bool, result: list of (x,y) atau error message)
        """
        success, cluster_result = EdgeClustering.cluster_edge_pixels(
            image_path, threshold, eps, min_samples
        )

        if not success:
            return False, cluster_result

        if use_noise:
            # Gunakan noise pixels (isolated edges)
            # Lebih aman karena edge yang terisolasi
            coords = cluster_result['noise_coords']
        else:
            # Gunakan clustered pixels (grouped edges)
            # Lebih banyak kapasitas
            coords = cluster_result['clustered_coords']

        if len(coords) == 0:
            msg = "noise" if use_noise else "clustered"
            return False, f"Tidak ada {msg} edge pixels ditemukan"

        return True, coords

    @staticmethod
    def visualize_clusters(image_path, output_path, threshold=50, eps=3, min_samples=5):
        """
        Visualisasi hasil clustering dengan warna berbeda per cluster

        Args:
            image_path: Path ke gambar
            output_path: Path untuk save visualisasi
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples

        Returns:
            tuple: (success: bool, message: str)
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
            colors = np.random.randint(50, 255, size=(n_clusters, 3))

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

            return True, f"Visualisasi cluster berhasil disimpan di: {output_path}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_capacity_clustered(image_path, threshold=50, eps=3, min_samples=5, use_noise=False):
        """
        Hitung kapasitas steganography dengan clustering

        Args:
            image_path: Path ke gambar
            threshold: Threshold edge detection
            eps: DBSCAN eps
            min_samples: DBSCAN min_samples
            use_noise: True = gunakan noise pixels, False = gunakan clustered pixels

        Returns:
            tuple: (success: bool, result: dict atau error message)
        """
        success, coords = EdgeClustering.get_optimized_edge_pixels(
            image_path, threshold, eps, min_samples, use_noise
        )

        if not success:
            return False, coords

        # Hitung kapasitas
        edge_count = len(coords)
        max_bits = edge_count * 3  # RGB = 3 bit per pixel
        max_chars = max_bits // 8 - 7  # Kurangi delimiter

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
            'use_noise': use_noise
        }

        return True, result
