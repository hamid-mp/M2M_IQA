import cv2
from imagecodecs import jpeg2k_encode
import numpy

array = numpy.random.randint(100, 200, (256, 256, 3), numpy.uint8)
encoded = jpeg2k_encode(array, level=0)
print(bytes(encoded[:12]))