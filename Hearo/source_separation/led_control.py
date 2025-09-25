from pixel_ring.pixel_ring.usb_pixel_ring_v2 import PixelRing
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

"""
Here are the usb_pixel_ring APIs.

Command	Data	API	Note
0	[0]	pixel_ring.trace()	trace mode, LEDs changing depends on VADand DOA
1	[red, green, blue, 0]	pixel_ring.mono()	mono mode, set all RGB LED to a single color, for example Red(0xFF0000), Green(0x00FF00)， Blue(0x0000FF)
2	[0]	pixel_ring.listen()	listen mode, similar with trace mode, but not turn LEDs off
3	[0]	pixel_ring.speak()	wait mode
4	[0]	pixel_ring.think()	speak mode
5	[0]	pixel_ring.spin()	spin mode
6	[r, g, b, 0] * 12	pixel_ring.customize()	custom mode, set each LED to its own color
0x20	[brightness]	pixel_ring.set_brightness()	set brightness, range: 0x00~0x1F
0x21	[r1, g1, b1, 0, r2, g2, b2, 0]	pixel_ring.set_color_palette()	set color palette, for example, pixel_ring.set_color_palette(0xff0000, 0x00ff00) together with pixel_ring.think()
0x22	[vad_led]	pixel_ring.set_vad_led()	set center LED: 0 - off, 1 - on, else - depends on VAD
0x23	[volume]	pixel_ring.set_volume()	show volume, range: 0 ~ 12
0x24	[pattern]	pixel_ring.change_pattern()	set pattern, 0 - Google Home pattern, others - Echo pattern
"""
