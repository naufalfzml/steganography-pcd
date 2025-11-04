import numpy as np
from PIL import Image


class EdgeDetection:
    """
    Kelas untuk melakukan edge detection menggunakan metode Sobel
    """

    @staticmethod
    def apply_sobel(image_array):
        """
        Menerapkan Sobel edge detection pada gambar

        Args:
            image_array: Array numpy dari gambar (grayscale)

        Returns:
            Array numpy dengan edge magnitude
        """
        # Sobel kernels
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        height, width = image_array.shape
        gradient_x = np.zeros_like(image_array, dtype=float)
        gradient_y = np.zeros_like(image_array, dtype=float)

        # Konvolusi dengan Sobel kernel
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Ambil neighborhood 3x3
                region = image_array[y-1:y+2, x-1:x+2]

                # Hitung gradient
                gradient_x[y, x] = np.sum(region * sobel_x)
                gradient_y[y, x] = np.sum(region * sobel_y)

        # Hitung magnitude
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        return magnitude

    @staticmethod
    def get_edge_pixels(image_path, threshold=50):
        """
        Mendapatkan koordinat pixel-pixel yang merupakan edge

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk menentukan edge (default: 50)

        Returns:
            tuple: (success: bool, result: list of (x, y) coordinates atau error message)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke grayscale untuk edge detection
            gray_img = img.convert('L')
            img_array = np.array(gray_img)

            # Terapkan Sobel
            edge_magnitude = EdgeDetection.apply_sobel(img_array)

            # Normalize ke range 0-255
            edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)

            # Ambil koordinat pixel yang merupakan edge (magnitude > threshold)
            edge_coords = []
            height, width = edge_magnitude.shape

            for y in range(height):
                for x in range(width):
                    if edge_magnitude[y, x] > threshold:
                        edge_coords.append((x, y))

            return True, edge_coords

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def visualize_edges(image_path, output_path, threshold=50):
        """
        Membuat visualisasi edge detection dan menyimpannya

        Args:
            image_path: Path ke gambar input
            output_path: Path untuk menyimpan hasil visualisasi
            threshold: Threshold untuk edge detection

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Buka gambar
            img = Image.open(image_path)

            # Convert ke grayscale
            gray_img = img.convert('L')
            img_array = np.array(gray_img)

            # Terapkan Sobel
            edge_magnitude = EdgeDetection.apply_sobel(img_array)

            # Normalize ke range 0-255
            edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)

            # Apply threshold
            edge_binary = np.where(edge_magnitude > threshold, 255, 0).astype(np.uint8)

            # Simpan hasil
            edge_img = Image.fromarray(edge_binary)
            edge_img.save(output_path)

            return True, f"Visualisasi edge berhasil disimpan di: {output_path}"

        except FileNotFoundError:
            return False, f"File '{image_path}' tidak ditemukan!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_edge_statistics(image_path, threshold=50):
        """
        Mendapatkan statistik edge dalam gambar

        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk edge detection

        Returns:
            tuple: (success: bool, stats: dict atau error message)
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            total_pixels = width * height

            success, edge_coords = EdgeDetection.get_edge_pixels(image_path, threshold)

            if not success:
                return False, edge_coords

            edge_count = len(edge_coords)
            edge_percentage = (edge_count / total_pixels) * 100

            # Hitung kapasitas untuk steganography
            # Setiap edge pixel bisa menyimpan 3 bit (RGB)
            max_bits = edge_count * 3
            max_chars = max_bits // 8 - 7  # Kurangi delimiter

            stats = {
                'total_pixels': total_pixels,
                'edge_pixels': edge_count,
                'edge_percentage': edge_percentage,
                'max_capacity_chars': max_chars,
                'threshold': threshold
            }

            return True, stats

        except Exception as e:
            return False, f"Error: {str(e)}"
