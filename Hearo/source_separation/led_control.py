from usb_pixel_ring_v2 import PixelRing
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
print(dev)
if dev:
    pixel_ring = PixelRing(dev)
    pixel_ring.wakeup(180)
    pixel_ring.set_brightness(0x001)

    while True:
        try:
            
            pixel_ring.mono(0x00FF00)
            pixel_ring.listen()
            
        
        except KeyboardInterrupt:
            break

    pixel_ring.off()