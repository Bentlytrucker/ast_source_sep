from usb_pixel_ring_v2 import PixelRing
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
print(dev)
if dev:
    pixel_ring = PixelRing(dev)
    
    pixel_ring.set_brightness(0x001)

    while True:
        try:
            time.sleep(5)
            pixel_ring.wakeup(180)
            pixel_ring.mono()	#mono mode, set all RGB LED to a single color, for example Red(0xFF0000), Green(0x00FF00)， Blue(0x0000FF)
        
        except KeyboardInterrupt:
            break

    pixel_ring.off()