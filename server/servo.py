import RPi.GPIO as GPIO
import time
import random as rand


SERVO_PINA = 18  # GPIO pin for the servo
SERVO_PINB = 19
# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PINA, GPIO.OUT)
GPIO.setup(SERVO_PINB, GPIO.OUT)
# Create PWM instance at 50Hz (Standard for SG90 servos)
pwmA = GPIO.PWM(SERVO_PINA, 50)
pwmB = GPIO.PWM(SERVO_PINB, 50)
#pwm.start(2.5)  # Start at 0° position
pwm=pwmA


def move_servo(times):

    pwm.start(7.5)
    for i in range(times):
        pwm.start(7.5)
        print(f"🌀 Moving Servo to 90° (Iteration {i + 1})")        
        pwm.ChangeDutyCycle(10)  # Move to 90°
        time.sleep(2)

        print(f"🔄 Moving Servo to 0° (Iteration {i + 1})")
        pwm.ChangeDutyCycle(5)  # Move to 0°
        time.sleep(2)

        print(f"🔄 Moving Servo to 0° (Iteration {i + 1})")
        pwm.ChangeDutyCycle(7.5)  # Move to 0°
        time.sleep(1)

    pwm.stop()
    print("✅ Servo movement completed.")

def move_2servos(times):

   
    for i in range(times):
        servo_choice=rand.randint(0,1)
        print(servo_choice)
        if servo_choice==0:
            pwm=pwmA
        else:
            pwm=pwmB
        pwm.start(7.5)
        print(f"🌀 Moving Servo to 90° (Iteration {i + 1})")        
        pwm.ChangeDutyCycle(10)  # Move to 90°
        time.sleep(2)

        print(f"🔄 Moving Servo to 0° (Iteration {i + 1})")
        pwm.ChangeDutyCycle(5)  # Move to 0°
        time.sleep(2)

        print(f"🔄 Moving Servo to 0° (Iteration {i + 1})")
        pwm.ChangeDutyCycle(7.5)  # Move to 0°
        time.sleep(1)

    pwm.stop()
    print("✅ Servo movement completed.")




if __name__ == "__main__":
    move_2servos(5)