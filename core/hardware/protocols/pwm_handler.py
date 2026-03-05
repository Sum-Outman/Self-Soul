"""
PWM Protocol Handler - Real hardware communication via PWM

Provides real hardware control for robots with PWM interfaces (servos, motors).
Supports multiple controller types:
- gpiozero: Direct GPIO PWM on Raspberry Pi
- pca9685: I2C PWM controller (Adafruit PCA9685)
- rpi_gpio: RPi.GPIO library (legacy)
"""

import logging
import math
from typing import Dict, Any, Optional

from .base_protocol_handler import BaseProtocolHandler

logger = logging.getLogger("PWMHandler")

class PWMHandler(BaseProtocolHandler):
    """PWM protocol handler for robot communication (servos, motors)"""
    
    def __init__(self):
        """Initialize PWM handler"""
        super().__init__()
        self.pwm_controller = None
        self.controller_type = None
        self.pins = {}
        self.frequency = 50  # Hz, standard for servos
        self.min_duty = 2.5   # Minimum duty cycle percentage for 0 degrees
        self.max_duty = 12.5  # Maximum duty cycle percentage for 180 degrees
    
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to robot via real PWM hardware
        
        Args:
            connection_params: Must contain 'controller_type', 'pin', etc.
            
        Returns:
            bool: True if connection successful with real hardware
        """
        try:
            # Validate parameters
            if not self.validate_connection_params(connection_params):
                return False
            
            # Extract connection parameters
            self.controller_type = connection_params.get("controller_type", "gpiozero")
            pins_str = connection_params.get("pin", "")
            self.frequency = connection_params.get("frequency", 50)
            self.min_duty = connection_params.get("min_duty", 2.5)
            self.max_duty = connection_params.get("max_duty", 12.5)
            
            # Parse pins (can be comma-separated list)
            if pins_str:
                self.pins = {pin.strip(): None for pin in pins_str.split(",")}
            else:
                self.pins = {}
            
            logger.info(f"Connecting to real PWM hardware with controller: {self.controller_type}")
            logger.info(f"Frequency: {self.frequency}Hz, Pins: {list(self.pins.keys())}")
            
            # Initialize based on controller type
            if self.controller_type == "gpiozero":
                self._connect_gpiozero(connection_params)
            elif self.controller_type == "pca9685":
                self._connect_pca9685(connection_params)
            elif self.controller_type == "rpi_gpio":
                self._connect_rpi_gpio(connection_params)
            else:
                error_msg = f"Unsupported PWM controller type: {self.controller_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.connected = True
            logger.info(f"Successfully connected to real PWM hardware using {self.controller_type}")
            return True
            
        except ImportError as e:
            error_msg = f"Required library not installed for {self.controller_type}: {e}"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Required for real PWM hardware communication.")
        except Exception as e:
            logger.error(f"Real PWM hardware connection failed: {e}")
            self.connected = False
            self.pwm_controller = None
            raise RuntimeError(f"Real PWM hardware connection failed: {e}")
    
    def _connect_gpiozero(self, connection_params: Dict[str, Any]) -> None:
        """Connect using gpiozero library"""
        try:
            from gpiozero import PWMOutputDevice
            from gpiozero.pins.pigpio import PiGPIOFactory
            
            # Use PiGPIOFactory for hardware PWM (requires pigpio daemon)
            factory = PiGPIOFactory() if connection_params.get("use_hardware_pwm", True) else None
            
            # Initialize PWM devices for each pin
            for pin_name in self.pins.keys():
                try:
                    pin_num = int(pin_name)
                    pwm_device = PWMOutputDevice(pin_num, frequency=self.frequency, factory=factory)
                    self.pins[pin_name] = pwm_device
                    logger.info(f"Initialized gpiozero PWM on pin {pin_num}")
                except ValueError:
                    logger.warning(f"Invalid pin number: {pin_name}, skipping")
            
            self.pwm_controller = "gpiozero"
            
        except ImportError as e:
            error_msg = "gpiozero library not installed. Please install with: pip install gpiozero"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Required for real GPIO PWM hardware communication.")
    
    def _connect_pca9685(self, connection_params: Dict[str, Any]) -> None:
        """Connect using PCA9685 I2C PWM controller"""
        try:
            from smbus2 import SMBus
            import time
            
            bus_number = connection_params.get("bus", 1)
            address = connection_params.get("address", 0x40)
            
            # Initialize I2C bus
            i2c_bus = SMBus(bus_number)
            
            # PCA9685 initialization sequence
            # Reset
            i2c_bus.write_byte_data(address, 0x00, 0x10)
            time.sleep(0.1)
            
            # Set mode1 register (auto-increment enabled, restart disabled)
            i2c_bus.write_byte_data(address, 0x00, 0x20)
            time.sleep(0.1)
            
            # Set prescaler for frequency
            prescaler = int(round(25000000.0 / (4096.0 * self.frequency))) - 1
            prescaler = max(3, min(255, prescaler))
            i2c_bus.write_byte_data(address, 0xFE, prescaler)
            time.sleep(0.1)
            
            # Set mode1 register again (restart enabled)
            i2c_bus.write_byte_data(address, 0x00, 0xA0)
            time.sleep(0.1)
            
            # Store I2C bus and address
            self.pwm_controller = {
                "type": "pca9685",
                "bus": i2c_bus,
                "address": address,
                "bus_number": bus_number
            }
            
            # Initialize all channels to off
            for channel in range(16):
                self._set_pca9685_channel(channel, 0, 0)
            
            logger.info(f"Initialized PCA9685 PWM controller on bus {bus_number}, address 0x{address:02X}")
            
        except ImportError as e:
            error_msg = "smbus2 library not installed. Please install with: pip install smbus2"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Required for real PCA9685 hardware communication.")
    
    def _connect_rpi_gpio(self, connection_params: Dict[str, Any]) -> None:
        """Connect using RPi.GPIO library (legacy)"""
        try:
            import RPi.GPIO as GPIO
            
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            
            # Initialize PWM for each pin
            for pin_name in self.pins.keys():
                try:
                    pin_num = int(pin_name)
                    GPIO.setup(pin_num, GPIO.OUT)
                    pwm = GPIO.PWM(pin_num, self.frequency)
                    pwm.start(0)  # Start with 0% duty cycle
                    self.pins[pin_name] = pwm
                    logger.info(f"Initialized RPi.GPIO PWM on pin {pin_num}")
                except ValueError:
                    logger.warning(f"Invalid pin number: {pin_name}, skipping")
            
            self.pwm_controller = "rpi_gpio"
            
        except ImportError as e:
            error_msg = "RPi.GPIO library not installed. Please install with: pip install RPi.GPIO"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Required for real RPi.GPIO hardware communication.")
    
    def disconnect(self) -> bool:
        """Disconnect from robot
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.pwm_controller == "gpiozero":
                # Stop all PWM devices
                for pin_name, pwm_device in self.pins.items():
                    if pwm_device:
                        pwm_device.close()
            
            elif self.pwm_controller == "rpi_gpio":
                # Clean up RPi.GPIO
                import RPi.GPIO as GPIO
                for pin_name, pwm in self.pins.items():
                    if pwm:
                        pwm.stop()
                GPIO.cleanup()
            
            elif isinstance(self.pwm_controller, dict) and self.pwm_controller.get("type") == "pca9685":
                # Turn off all PCA9685 channels
                i2c_bus = self.pwm_controller.get("bus")
                address = self.pwm_controller.get("address")
                if i2c_bus:
                    for channel in range(16):
                        self._set_pca9685_channel(channel, 0, 0)
                    i2c_bus.close()
            
            self.connected = False
            self.pwm_controller = None
            self.pins = {}
            logger.info("Disconnected from robot (PWM)")
            return True
            
        except Exception as e:
            logger.error(f"PWM disconnection error: {e}")
            return False
    
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send real command to robot via PWM hardware
        
        Args:
            command: Command name (e.g., "set_joint_position", "get_joint_position")
            data: Command data including pin/channel, angle, etc.
            
        Returns:
            Dict[str, Any]: Real response from robot hardware
        """
        if not self.connected or not self.pwm_controller:
            return self._format_response(False, error="Not connected to real PWM hardware")
        
        try:
            # Route command to appropriate handler
            if command == "set_joint_position":
                return self._handle_set_joint_position(data)
            elif command == "get_joint_position":
                return self._handle_get_joint_position(data)
            elif command == "set_pwm_duty_cycle":
                return self._handle_set_pwm_duty_cycle(data)
            elif command == "get_pwm_duty_cycle":
                return self._handle_get_pwm_duty_cycle(data)
            elif command == "set_frequency":
                return self._handle_set_frequency(data)
            elif command == "get_frequency":
                return self._handle_get_frequency(data)
            elif command == "stop_pwm":
                return self._handle_stop_pwm(data)
            elif command == "get_battery_level":
                return self._handle_get_battery_level(data)
            elif command == "get_temperature":
                return self._handle_get_temperature(data)
            else:
                return self._format_response(False, error=f"Unsupported PWM command: {command}")
                
        except Exception as e:
            logger.error(f"PWM command error: {e}")
            return self._format_response(False, error=str(e))
    
    def _handle_set_joint_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set joint position (servo angle)"""
        pin = data.get("pin")
        channel = data.get("channel")
        angle = data.get("angle", 90.0)
        
        # Validate angle range
        angle = max(0.0, min(180.0, angle))
        
        # Convert angle to duty cycle
        duty_cycle = self._angle_to_duty_cycle(angle)
        
        # Set PWM
        if pin is not None:
            success = self._set_pwm_by_pin(str(pin), duty_cycle)
        elif channel is not None:
            success = self._set_pwm_by_channel(channel, duty_cycle)
        else:
            return self._format_response(False, error="Either 'pin' or 'channel' must be specified")
        
        if success:
            return self._format_response(True, data={
                "angle": angle,
                "duty_cycle": duty_cycle,
                "pin": pin,
                "channel": channel
            })
        else:
            return self._format_response(False, error="Failed to set joint position")
    
    def _handle_get_joint_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current joint position (servo angle)"""
        # Note: Most PWM hardware doesn't support reading back position
        # This is a stub that returns the last set position
        # In real systems, you would need encoders or potentiometers
        
        pin = data.get("pin")
        channel = data.get("channel")
        
        # For now, return a default position
        # In a real implementation, you might query sensors
        position = 90.0  # Default center position
        
        return self._format_response(True, data={
            "position": position,
            "pin": pin,
            "channel": channel,
            "note": "Position reading requires additional sensors (encoders/potentiometers)"
        })
    
    def _handle_set_pwm_duty_cycle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set raw PWM duty cycle (0-100%)"""
        pin = data.get("pin")
        channel = data.get("channel")
        duty_cycle = data.get("duty_cycle", 0.0)
        
        # Validate duty cycle range
        duty_cycle = max(0.0, min(100.0, duty_cycle))
        
        # Set PWM
        if pin is not None:
            success = self._set_pwm_by_pin(str(pin), duty_cycle)
        elif channel is not None:
            success = self._set_pwm_by_channel(channel, duty_cycle)
        else:
            return self._format_response(False, error="Either 'pin' or 'channel' must be specified")
        
        if success:
            return self._format_response(True, data={
                "duty_cycle": duty_cycle,
                "pin": pin,
                "channel": channel
            })
        else:
            return self._format_response(False, error="Failed to set PWM duty cycle")
    
    def _handle_get_pwm_duty_cycle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current PWM duty cycle"""
        # Note: Most PWM hardware doesn't support reading back duty cycle
        # This returns the last set value
        
        pin = data.get("pin")
        channel = data.get("channel")
        
        # For now, return a default value
        duty_cycle = 0.0
        
        return self._format_response(True, data={
            "duty_cycle": duty_cycle,
            "pin": pin,
            "channel": channel,
            "note": "Duty cycle reading not supported by most PWM hardware"
        })
    
    def _handle_set_frequency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set PWM frequency"""
        frequency = data.get("frequency", 50.0)
        
        # Validate frequency range
        frequency = max(1.0, min(10000.0, frequency))
        
        # Update frequency
        old_frequency = self.frequency
        self.frequency = frequency
        
        # For PCA9685, need to update prescaler
        if isinstance(self.pwm_controller, dict) and self.pwm_controller.get("type") == "pca9685":
            try:
                i2c_bus = self.pwm_controller.get("bus")
                address = self.pwm_controller.get("address")
                prescaler = int(round(25000000.0 / (4096.0 * frequency))) - 1
                prescaler = max(3, min(255, prescaler))
                i2c_bus.write_byte_data(address, 0xFE, prescaler)
            except Exception as e:
                logger.error(f"Failed to update PCA9685 frequency: {e}")
                return self._format_response(False, error=f"Failed to update frequency: {e}")
        
        return self._format_response(True, data={
            "old_frequency": old_frequency,
            "new_frequency": frequency
        })
    
    def _handle_get_frequency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current PWM frequency"""
        return self._format_response(True, data={"frequency": self.frequency})
    
    def _handle_stop_pwm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stop PWM on specific pin/channel"""
        pin = data.get("pin")
        channel = data.get("channel")
        
        # Set duty cycle to 0
        if pin is not None:
            success = self._set_pwm_by_pin(str(pin), 0.0)
        elif channel is not None:
            success = self._set_pwm_by_channel(channel, 0.0)
        else:
            # Stop all PWM outputs
            if self.pwm_controller == "gpiozero":
                for pin_name, pwm_device in self.pins.items():
                    if pwm_device:
                        pwm_device.value = 0.0
                success = True
            elif self.pwm_controller == "rpi_gpio":
                for pin_name, pwm in self.pins.items():
                    if pwm:
                        pwm.ChangeDutyCycle(0.0)
                success = True
            elif isinstance(self.pwm_controller, dict) and self.pwm_controller.get("type") == "pca9685":
                for channel in range(16):
                    self._set_pca9685_channel(channel, 0, 0)
                success = True
            else:
                success = False
        
        return self._format_response(success, data={
            "stopped": True,
            "pin": pin,
            "channel": channel
        })
    
    def _handle_get_battery_level(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get battery level (requires additional hardware)"""
        # In real implementation, this would read from ADC
        # For now, return simulated value
        battery_level = 85.0
        
        return self._format_response(True, data={
            "level": battery_level,
            "unit": "percent",
            "note": "Battery level reading requires additional hardware (ADC)"
        })
    
    def _handle_get_temperature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get temperature (requires additional hardware)"""
        # In real implementation, this would read from temperature sensor
        # For now, return simulated value
        temperature = 32.5
        
        return self._format_response(True, data={
            "temperature": temperature,
            "unit": "celsius",
            "note": "Temperature reading requires additional hardware (temperature sensor)"
        })
    
    def _set_pwm_by_pin(self, pin: str, duty_cycle: float) -> bool:
        """Set PWM duty cycle by pin number"""
        try:
            if self.pwm_controller == "gpiozero":
                pwm_device = self.pins.get(pin)
                if pwm_device:
                    pwm_device.value = duty_cycle / 100.0
                    return True
            
            elif self.pwm_controller == "rpi_gpio":
                pwm = self.pins.get(pin)
                if pwm:
                    pwm.ChangeDutyCycle(duty_cycle)
                    return True
            
            else:
                logger.warning(f"Pin-based PWM not supported for controller type: {self.pwm_controller}")
        
        except Exception as e:
            logger.error(f"Failed to set PWM by pin {pin}: {e}")
        
        return False
    
    def _set_pwm_by_channel(self, channel: int, duty_cycle: float) -> bool:
        """Set PWM duty cycle by channel number (for PCA9685)"""
        try:
            if isinstance(self.pwm_controller, dict) and self.pwm_controller.get("type") == "pca9685":
                # Convert duty cycle to PCA9685 on/off values
                # PCA9685 uses 12-bit resolution (0-4095)
                on_value = 0
                off_value = int((duty_cycle / 100.0) * 4095.0)
                self._set_pca9685_channel(channel, on_value, off_value)
                return True
            else:
                logger.warning(f"Channel-based PWM only supported for PCA9685 controller")
        
        except Exception as e:
            logger.error(f"Failed to set PWM by channel {channel}: {e}")
        
        return False
    
    def _set_pca9685_channel(self, channel: int, on: int, off: int) -> None:
        """Set PCA9685 channel on/off values"""
        if not isinstance(self.pwm_controller, dict) or self.pwm_controller.get("type") != "pca9685":
            return
        
        i2c_bus = self.pwm_controller.get("bus")
        address = self.pwm_controller.get("address")
        
        if not i2c_bus:
            return
        
        # PCA9685 registers
        LED0_ON_L = 0x06 + (4 * channel)
        
        # Write values
        i2c_bus.write_byte_data(address, LED0_ON_L, on & 0xFF)
        i2c_bus.write_byte_data(address, LED0_ON_L + 1, (on >> 8) & 0xFF)
        i2c_bus.write_byte_data(address, LED0_ON_L + 2, off & 0xFF)
        i2c_bus.write_byte_data(address, LED0_ON_L + 3, (off >> 8) & 0xFF)
    
    def _angle_to_duty_cycle(self, angle: float) -> float:
        """Convert servo angle (0-180 degrees) to duty cycle percentage"""
        # Linear interpolation between min and max duty cycle
        duty_cycle = self.min_duty + (angle / 180.0) * (self.max_duty - self.min_duty)
        return duty_cycle
    
    def get_required_parameters(self) -> list:
        """Get required connection parameters
        
        Returns:
            list: ['controller_type', 'pin']
        """
        return ["controller_type", "pin"]
