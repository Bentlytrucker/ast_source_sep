from ./usb_4_mic_array/tuning import Tuning
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
#print dev
if dev:
    Mic_tuning = Tuning(dev)
    Mic_tuning.set_vad_threshold(1.8)
    print(Mic_tuning.is_voice())

    while True:
        try:
            print(Mic_tuning.is_voice())
            time.sleep(1)
        except KeyboardInterrupt:
            break