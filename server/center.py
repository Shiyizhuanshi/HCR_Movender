import eventlet

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import RPi.GPIO as GPIO
import time
from flask_cors import CORS
import servo 
import payment

import json
import threading
from websocket import create_connection
import argparse
from controller import GamepadController
import cvclass
import movender
import os


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

port = 5001


# In server.py
state = {
    "engaged": False,
    "goal_pos": None,
    "scores": []
}


@app.route("/")
def home():
    return "Simple Messaging Server is Running"

@socketio.on("send_checkout")
def handle_send_payment():
    print("payment request received")
    socketio.emit("payment", {"checkout": 1})
    return {"message": "Checkout voice command received and processed.", "checkout": 1}

# Handle payment requests
@socketio.on("send_payment")
def handle_send_payment(data):
    quantity = data.get("quantity", 0)
    if int(quantity and payment.poll_nfc() == 1) > 0:
        print(f"✅ Received Payment Success via WebSocket, quantity: {quantity}")
        print("Sending payment success to WebSocket clients...")
        socketio.emit("payment", {"success": 1})
        os.system("aplay audios/thanks.wav")
        servo.move_2servos(quantity)
        
        
        return {"message": "Payment received and processed.", "success": 1, "quantity": quantity}
    else:
        print("⚠️ No valid success or quantity parameter received via WebSocket.")
        return {"error": "No valid parameters provided."}


def auto_send_voice_pos():
    prev_goal = None
    prev_state = state.get("engaged")
    while True:
        # Get the current goal position from the shared state
        current_goal = state.get("goal_pos")
        cur_state = state.get("engaged")

        # Check if the goal has just become set
        if prev_goal is None and current_goal is not None:
            send_goal_to_ros(current_goal)
        prev_goal = current_goal
        
        if prev_state != cur_state:
            socketio.emit("voice", {
                "engaged": cur_state,
                "score": state.get("score")
            })
            prev_state = cur_state
        eventlet.sleep(5)


@app.route("/hhh", methods=["GET"])
def print_global():
    return jsonify({
        "engaged": state.get("engaged"),
        "goal_pos": state.get("goal_pos"),
        "score": state.get("score")
    })

@app.route("/confirm_purchase", methods=["GET"])
def confirm_purchase():
    quantity = request.args.get("quantity", type=int)
    if quantity and quantity > 0:
        socketio.emit("voice", {"success": 1})
        print(f"✅ Received Purchase Quantity: {quantity}")
        return f"Quantity: {quantity} received", 200
    else:
        return "No quantity received.", 400

# WebSocket Connection Handling
@socketio.on("connect")
def handle_connect():
    print("New WebSocket client connected")

@socketio.on("connect1")
def handle_connect1():
    print("11111111111111111111111111111111111")

@socketio.on("disconnect")
def handle_disconnect():
    print("WebSocket client disconnected")

rosbridge_ws = None
last_x = 0
last_y = 0
last_event_time = 0
update_frequency = 0.1

def connect_to_rosbridge():
    global rosbridge_ws
    try:
        rosbridge_ws = create_connection("ws://localhost:9090/")
        print("✅ Connected to rosbridge websocket.")

        # Advertise /cmd_vel
        advertise_cmd_msg = {
            "op": "advertise",
            "topic": "/RosAria/cmd_vel",
            "type": "geometry_msgs/Twist"
        }
        rosbridge_ws.send(json.dumps(advertise_cmd_msg))
        print("✅ Advertised topic /cmd_vel with type geometry_msgs/Twist.")

        # Advertise /goal_pos
        advertise_goal_msg = {
            "op": "advertise",
            "topic": "/goal_pos",
            "type": "std_msgs/String"
        }
        rosbridge_ws.send(json.dumps(advertise_goal_msg))
        print("✅ Advertised topic /goal_pos with type std_msgs/String.")

    except Exception as e:
        rosbridge_ws = None
        print("⚠️ Failed to connect to rosbridge:", e)

MAX_SPEED = 1.0

def publish_twist():
    global rosbridge_ws, last_x, last_y
    if rosbridge_ws is None:
        print("⚠️ rosbridge_ws not connected.")
        return

    linear_speed = -(last_y / 255.0) * MAX_SPEED
    angular_speed = -(last_x / 255.0) * MAX_SPEED

    linear_speed = max(-MAX_SPEED, min(MAX_SPEED, linear_speed))
    angular_speed = max(-MAX_SPEED, min(MAX_SPEED, angular_speed))

    twist_msg = {
        "op": "publish",
        "topic": "/RosAria/cmd_vel",
        "msg": {
            "linear": {"x": linear_speed, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": angular_speed}
        }
    }

    try:
        rosbridge_ws.send(json.dumps(twist_msg))
    except Exception as e:
        print("⚠️ Failed to publish twist:", e)

def controller_callback(event_type, data):
    global last_x, last_y, last_event_time, update_frequency
    current_time = time.time()
    if current_time - last_event_time < update_frequency:
        return
    last_event_time = current_time

    if event_type == "ABS_X":
        last_x = data
    elif event_type == "ABS_Y":
        last_y = data

    publish_twist()

def start_controller_listener():
    try:
        controller = GamepadController()
        print("Starting GamepadController listener on device:", controller.gamepad.name)
        controller.register_listener(controller_callback)
    except Exception as e:
        print("Error starting controller listener:", e)

def send_goal_to_ros(goal_pos):
    message = str(goal_pos[0])+","+str(goal_pos[1])
    msg = {
        "op": "publish",
        "topic": "/goal_pos",
        "msg": {
            "data": message,
        }
    }
    try:
        rosbridge_ws=create_connection("ws://localhost:9090/")
        print("✅ Connected to rosbridge websocket.")

        rosbridge_ws.send(json.dumps(msg))
        print("✅ Sent goal position to ROS:", goal_pos)
    except Exception as e:
        print("⚠️ Failed to send goal position to ROS:", e)



def start_controller_and_rosbridge():
    connect_to_rosbridge()
    start_controller_listener()

def run_detector(state):
    detector = cvclass.CVGoalDetector(state=state, display=False)
    detector.run()

if __name__ == "__main__":
    connect_to_rosbridge()

    state = {
        "engaged": False,
        "goal_pos": None,
        "scores": []
    }

    bot = movender.Movender()
    movender_thread = threading.Thread(target=bot.start, daemon=True)
    movender_thread.start()
    
    # Start the CV detector in its own daemon thread.
    detector_thread = threading.Thread(target=run_detector, args=(state,))
    detector_thread.daemon = True
    detector_thread.start()    

    parser = argparse.ArgumentParser(description="Simple Messaging Server with optional controller support")
    parser.add_argument("-control", action="store_true", help="Enable remote control using GamepadController")
    parser.add_argument("-update_frequency", type=float, default=0.1,
                        help="Minimum time interval (in seconds) between processing controller events (default: 0.1)")
    args = parser.parse_args()

    update_frequency = args.update_frequency
    print(f"Controller update frequency set to: {update_frequency} seconds")

    if args.control:
        print("Remote control enabled.")
        controller_thread = threading.Thread(target=start_controller_and_rosbridge)
        controller_thread.daemon = True
        controller_thread.start()
    else:
        print("Remote control not enabled.")

    # Start the background task before calling socketio.run()
    socketio.start_background_task(auto_send_voice_pos)
    print("Background task started.")

    try:
        print(f"Server is running on http://0.0.0.0:{port}")
        socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        print("🛑 Stopping Server...")
        
    
