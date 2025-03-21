import os
from py532lib.i2c import Pn532_i2c
import time

# ✅ Initialize NFC Reader
pn532 = Pn532_i2c()
pn532.SAMconfigure()

# ✅ NFC Polling Function with Timeout
def poll_nfc(timeout=30):
    """
    Polls NFC reader for up to `timeout` seconds.
    - Returns `1` if a card is detected.
    - Returns `0` if timeout expires with no card detected.
    """
    start_time = time.time()  # Record start time

    while time.time() - start_time < timeout:
        try:
            card_data = pn532.read_mifare().get_data()  # Read card data
            print(card_data)

            if card_data:
                os.system("aplay /home/hcr/Desktop/HCR_server/new/audios/di.wav")  # Play sound
                card_uid_hex = ":".join(f"{x:02X}" for x in card_data)  # Convert to hex
                print(f"✅ NFC Card Detected! UID: {card_uid_hex}")
                return 1  # ✅ Return success immediately

        except Exception:
            pass  # Ignore errors if no card is detected

    print("⏳ NFC polling timed out (No card detected).")
    return 0  

# ✅ Example Usage:
if __name__ == "__main__":
    result = poll_nfc(timeout=30)  # Run for 30 seconds
    print(f"NFC Polling Result: {result}")  # Output: 1 (if card detected), 0 (if timeout)
