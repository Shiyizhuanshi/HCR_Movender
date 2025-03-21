from flask import Flask, request, jsonify
import RPi.GPIO as GPIO
import time

app = Flask(__name__)

SERVO_PIN = 18  # GPIO pin for the servo

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance at 50Hz (Standard for SG90 servos)
pwm = GPIO.PWM(SERVO_PIN, 50)
#pwm.start(2.5)  # Start at 0° position



@app.route("/confirm_purchase", methods=["GET"])
def confirm_purchase():
    quantity = request.args.get("quantity", type=int)

    if quantity and quantity > 0:
        print(f"✅ Received Purchase Quantity: {quantity}")
        move_servo(quantity)
        return jsonify({"status": "success", "message": f"Servo moved {quantity} times."}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid quantity provided."}), 400

if __name__ == "__main__":
    try:
        print("🚀 Flask Servo Server is Running...")
        app.run(host="0.0.0.0", port=5001)
    except KeyboardInterrupt:
        print("🛑 Stopping Server...")
    finally:
        pwm.stop()
        GPIO.cleanup()
