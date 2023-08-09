import numpy as np
from PIL import Image

# Tạo mảng NumPy và hình ảnh PIL (mẫu)
numpy_array = np.random.rand(100, 100, 3) * 255  # Mảng ngẫu nhiên
pil_image = Image.fromarray(numpy_array.astype('uint8'))

# Kiểm tra xem mảng NumPy và hình ảnh PIL có giống nhau hay không
if np.array_equal(numpy_array, np.array(pil_image)):
    print("Giống nhau")
else:
    print("Khác nhau")
