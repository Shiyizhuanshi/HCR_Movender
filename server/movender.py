import os
import json
import time
import socketio
import threading
import subprocess
from vosk import Model, KaldiRecognizer
import pyaudio

class Movender:
    def __init__(self, model_path="model", device_index=1, server_url="http://localhost:5001"):
        # 配置
        self.model_path = model_path
        self.device_index = device_index
        self.server_url = server_url
        
        # 预设的 ASR 触发关键词及其音频文件
        self.predefined_responses = {
            "hello": "audios/hello.wav",
            "how are you": "audios/hello.wav",
            "what are you": "audios/intro.wav",
            "who are you": "audios/intro.wav",
            "good": "audios/intro.wav",
            "fine": "audios/intro.wav",
            "yes": "audios/payment.wav",
            "yeah": "audios/payment.wav",
            "sure": "audios/payment.wav",
            "ok": "audios/payment.wav",
            "okay": "audios/payment.wav",
            "thank you": "audios/thanks.wav",
            "fuck off": "audios/mamba.wav",
            "no": "audios/goodbye.wav",
            "don't": "audios/goodbye.wav",
            "sorry": "audios/goodbye.wav",
        }
        
        # 播放模式和控制
        self.engaged_mode = False
        self.shout_active = threading.Event()
        self.shout_process = None
        
        # 初始化 ASR
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                  frames_per_buffer=8192, input_device_index=self.device_index)
        self.stream.start_stream()
        
        self.last_recognition_time = time.time()
        self.currently_speaking = False
        
        # 初始化 Socket.IO 客户端
        self.sio = socketio.Client()
        self.setup_socket_events()
        
    def setup_socket_events(self):
        @self.sio.event
        def connect():
            print("✅ Connected to server!")
        
        @self.sio.event
        def disconnect():
            print("❌ Disconnected from server!")
        
        @self.sio.on("voice")
        def handle_server_response(data):
            print(f"📩 Server Response: {data}")
            
            if "success" in data and data["success"] == 1:
                self.speak("thanks.wav")
                return
            
            if "engaged" in data:
                self.engaged_mode = data["engaged"]
                if self.engaged_mode:
                    self.shout_active.set()
                    if self.shout_process and self.shout_process.poll() is None:
                        print("🛑 Stopping shout_slow.wav and switching to ASR mode...")
                        self.shout_process.terminate()
                        self.shout_process.wait()
                        self.shout_process = None
                    if not self.stream.is_active():
                        print("🎤 Restarting microphone for ASR...")
                        self.stream.start_stream()
                else:
                    self.shout_active.clear()
    
    def connect_to_server(self):
        print(f"🔗 Connecting to {self.server_url}...")
        while True:
            try:
                self.sio.connect(self.server_url)
                break
            except socketio.exceptions.ConnectionError:
                print("⚠️ Connection failed. Retrying in 2s...")
                time.sleep(2)
    
    def speak(self, audio_file):
        print(f"🔊 Playing: {audio_file}")
        if not os.path.exists(audio_file):
            print(f"⚠️ Warning: {audio_file} not found!")
            return
        
        self.currently_speaking = True
        if self.stream.is_active():
            self.stream.stop_stream()
        
        subprocess.run(["aplay", "-q", audio_file])
        time.sleep(0.2)
        
        if not self.stream.is_active():
            self.stream.start_stream()
        
        self.currently_speaking = False
    
    def shout_loop(self):
        while True:
            if not self.engaged_mode:
                print("📢 Shouting: shout_slow.wav")
                self.shout_process = subprocess.Popen(["aplay", "-q", "audios/shout_slow.wav"])
                self.shout_process.wait()
                for _ in range(10):
                    if self.shout_active.is_set():
                        print("🛑 Stopping shout mode...")
                        return
                    time.sleep(0.2)
            else:
                time.sleep(0.2)
    
    def asr_loop(self):
        print("🎤 Listening for customer input...")
        while True:
            if self.engaged_mode:
                if not self.stream.is_active():
                    print("🎤 Restarting microphone for ASR...")
                    self.stream.start_stream()
                
                data = self.stream.read(4096, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    recognized_text = result["text"]
                    
                    if time.time() - self.last_recognition_time > 3 and recognized_text:
                        self.last_recognition_time = time.time()
                        print(f"🎙️ Recognized: {recognized_text}")
                        
                        for keyword, audio_file in self.predefined_responses.items():
                            if keyword in recognized_text.lower():
                                self.speak(audio_file)
                                break
            else:
                time.sleep(0.2)
    
    def start(self):
        self.connect_to_server()
        threading.Thread(target=self.shout_loop, daemon=True).start()
        threading.Thread(target=self.asr_loop, daemon=True).start()

        # 主线程等待，保持程序运行（可根据需求调整）
        while True:
            time.sleep(1)


# 运行 Movender
if __name__ == "__main__":
    bot = Movender()
    bot.start()
