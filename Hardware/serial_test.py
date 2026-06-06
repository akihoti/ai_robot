# songjia_gimbal_control.py
# 适配松甲科技云台（Arduino/STM32控制板，270°PWM舵机）
# 支持：二维云台（ID0水平+ID1垂直）+ 一维云台（ID2）
import serial
import time
import serial.tools.list_ports
import sys
import platform

class SongJiaGimbal:
    """
    松甲云台控制器：严格遵循手册指令协议
    核心参数：波特率115200，PWM范围500-2500（对应0-270°）
    舵机ID分配：0=二维云台水平，1=二维云台垂直，2=一维云台
    """
    def __init__(self):
        self.ser = None
        self.is_connected = False
        # 舵机默认参数（手册标准）
        self.servo_ids = [0, 1, 2]  # 0=水平，1=垂直，2=一维云台
        self.min_pwm = 500       # 0°对应PWM
        self.max_pwm = 2500      # 270°对应PWM
        self.mid_pwm = 1500      # 135°中位对应PWM

    def list_available_ports(self):
        """列出所有可用串口（手册要求先确认串口）"""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("未检测到串口设备，请检查USB连接和CH340驱动")
            return []
        print("\n可用串口列表：")
        for i, port in enumerate(ports):
            print(f"  [{i+1}] {port.device} - {port.description}")
        return [port.device for port in ports]

    def connect(self, port=None, baudrate=115200):
        """
        连接云台（手册指定波特率115200）
        port: 串口路径，Linux下一般为/dev/ttyUSB0，Windows下为COMx
        若不指定则自动检测第一个可用串口
        """
        if port is None:
            ports = self.list_available_ports()
            if not ports:
                return False
            # 优先选择ttyUSB设备（Linux）
            usb_ports = [p for p in ports if 'USB' in p or 'ACM' in p]
            port = usb_ports[0] if usb_ports else ports[0]

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                bytesize=8,
                parity='N',  # 无校验（手册默认）
                stopbits=1   # 1位停止位（手册默认）
            )
            time.sleep(2)  # 等待控制板初始化（手册要求上电后延迟）
            self.is_connected = True
            print(f"\n成功连接云台：{port}（波特率：{baudrate}）")
            # 读取控制板反馈（验证连接）
            self.read_feedback()
            return True
        except serial.SerialException as e:
            if "Permission" in str(e) or "denied" in str(e):
                print(f"\n串口权限不足：{str(e)}")
                print("请尝试：sudo usermod -aG dialout $USER && 重新登录")
            else:
                print(f"\n连接失败：{str(e)}")
            return False
        except Exception as e:
            print(f"\n连接失败：{str(e)}")
            return False

    def read_feedback(self):
        """读取控制板串口反馈（手册指令执行后需确认响应）"""
        if not self.is_connected:
            return
        feedback = []
        time.sleep(0.1)  # 等待数据到达
        while self.ser.in_waiting > 0:
            try:
                msg = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if msg:
                    feedback.append(msg)
                    print(f"  控制板反馈：{msg}")
            except:
                break
        return feedback

    def angle_to_pwm(self, angle):
        """角度转PWM（手册标准：0-270°对应500-2500PWM）"""
        angle = max(0, min(angle, 270))
        pwm = self.min_pwm + (angle / 270) * (self.max_pwm - self.min_pwm)
        return int(pwm)

    def pwm_to_angle(self, pwm):
        """PWM转回角度（反向验证）"""
        pwm = max(self.min_pwm, min(pwm, self.max_pwm))
        angle = ((pwm - self.min_pwm) / (self.max_pwm - self.min_pwm)) * 270
        return round(angle, 1)

    def _get_servo_name(self, servo_id):
        """获取舵机名称"""
        names = {0: "二维云台-水平", 1: "二维云台-垂直", 2: "一维云台"}
        return names.get(servo_id, f"舵机{servo_id}")

    def move_single_servo(self, servo_id, angle, move_time=1000):
        """
        单个舵机转动（手册指令：#IndexPpwmTtime!）
        servo_id: 0=二维云台水平，1=二维云台垂直，2=一维云台
        angle: 目标角度（0-270°）
        move_time: 转动时间（0000-9999毫秒，手册要求4位数字）
        """
        if not self.is_connected:
            print("未连接云台，请先调用connect()")
            return False
        if servo_id not in self.servo_ids:
            print(f"舵机ID错误，支持ID：{self.servo_ids}")
            return False

        target_pwm = self.angle_to_pwm(angle)
        cmd = f"#{'%03d' % servo_id}P{'%04d' % target_pwm}T{'%04d' % move_time}!"
        servo_name = self._get_servo_name(servo_id)

        try:
            print(f"\n发送指令：{cmd}（{servo_name}转到{angle}°，耗时{move_time}ms）")
            self.ser.write(cmd.encode('utf-8'))
            time.sleep(move_time / 1000)
            self.read_feedback()
            return True
        except Exception as e:
            print(f"指令发送失败：{str(e)}")
            return False

    def move_both_2d_servos(self, horizontal_angle, vertical_angle, move_time=1000):
        """
        二维云台双舵机联动（手册指令：{#000PxxxTxxx!#001PxxxTxxx!}）
        horizontal_angle: 水平舵机目标角度
        vertical_angle: 垂直舵机目标角度
        """
        if not self.is_connected:
            print("未连接云台，请先调用connect()")
            return False

        h_pwm = self.angle_to_pwm(horizontal_angle)
        v_pwm = self.angle_to_pwm(vertical_angle)
        cmd = f"{{#000P{'%04d' % h_pwm}T{'%04d' % move_time}!#001P{'%04d' % v_pwm}T{'%04d' % move_time}!}}"

        try:
            print(f"\n发送联动指令：{cmd}（水平{horizontal_angle}°，垂直{vertical_angle}°）")
            self.ser.write(cmd.encode('utf-8'))
            time.sleep(move_time / 1000)
            self.read_feedback()
            return True
        except Exception as e:
            print(f"联动指令失败：{str(e)}")
            return False

    def move_all_servos(self, horizontal_angle, vertical_angle, oned_angle, move_time=1000):
        """
        全部舵机联动（二维云台+一维云台）
        手册指令：{#000PxxxTxxx!#001PxxxTxxx!#002PxxxTxxx!}
        """
        if not self.is_connected:
            print("未连接云台，请先调用connect()")
            return False

        h_pwm = self.angle_to_pwm(horizontal_angle)
        v_pwm = self.angle_to_pwm(vertical_angle)
        o_pwm = self.angle_to_pwm(oned_angle)
        cmd = f"{{#000P{'%04d' % h_pwm}T{'%04d' % move_time}!#001P{'%04d' % v_pwm}T{'%04d' % move_time}!#002P{'%04d' % o_pwm}T{'%04d' % move_time}!}}"

        try:
            print(f"\n发送全联动指令：{cmd}（水平{horizontal_angle}°，垂直{vertical_angle}°，一维{oned_angle}°）")
            self.ser.write(cmd.encode('utf-8'))
            time.sleep(move_time / 1000)
            self.read_feedback()
            return True
        except Exception as e:
            print(f"全联动指令失败：{str(e)}")
            return False

    def reset_to_mid(self, move_time=1000):
        """所有云台复位到中位（135°，手册标准中位）"""
        print("\n所有云台复位到中位（135°）")
        return self.move_all_servos(135, 135, 135, move_time)

    def stop_all_servos(self):
        """停止所有舵机（手册指令：$DST!）"""
        if not self.is_connected:
            print("未连接云台")
            return False
        cmd = "$DST!"
        print(f"\n发送停止指令：{cmd}")
        self.ser.write(cmd.encode('utf-8'))
        self.read_feedback()
        return True

    def stop_single_servo(self, servo_id):
        """停止单个舵机（手册指令：$DST:x!）"""
        if not self.is_connected:
            print("未连接云台")
            return False
        if servo_id not in self.servo_ids:
            print(f"舵机ID错误，支持ID：{self.servo_ids}")
            return False
        cmd = f"$DST:{servo_id}!"
        servo_name = self._get_servo_name(servo_id)
        print(f"\n发送停止指令：{cmd}（停止{servo_name}）")
        self.ser.write(cmd.encode('utf-8'))
        self.read_feedback()
        return True

    def disconnect(self):
        """断开串口连接"""
        if self.is_connected and self.ser:
            self.ser.close()
            self.is_connected = False
            print("\n已断开云台连接")

# -------------------------- 测试代码（直接运行即可） --------------------------
if __name__ == "__main__":
    gimbal = SongJiaGimbal()

    # 1. 连接云台（自动检测串口，也可手动指定 port="/dev/ttyUSB0"）
    if not gimbal.connect():
        sys.exit(1)

    try:
        # 2. 测试二维云台单舵机转动
        gimbal.move_single_servo(servo_id=0, angle=90, move_time=800)   # 二维云台水平转90°
        time.sleep(0.5)
        gimbal.move_single_servo(servo_id=1, angle=45, move_time=800)   # 二维云台垂直转45°
        time.sleep(0.5)

        # 3. 测试一维云台转动
        gimbal.move_single_servo(servo_id=2, angle=90, move_time=800)   # 一维云台转90°
        time.sleep(0.5)

        # 4. 测试二维云台双舵机联动
        gimbal.move_both_2d_servos(horizontal_angle=180, vertical_angle=90, move_time=1000)
        time.sleep(0.5)

        # 5. 测试全部舵机联动（二维+一维）
        gimbal.move_all_servos(horizontal_angle=90, vertical_angle=90, oned_angle=90, move_time=1000)
        time.sleep(0.5)

        # 6. 测试角度-PWM转换（手册标准验证）
        test_angle = 67.5
        test_pwm = gimbal.angle_to_pwm(test_angle)
        print(f"\n转换验证：{test_angle}° -> PWM:{test_pwm} -> 转回角度:{gimbal.pwm_to_angle(test_pwm)}°")

        # 7. 复位到中位
        gimbal.reset_to_mid(move_time=1000)

    except KeyboardInterrupt:
        print("\n\n用户终止操作，正在停止舵机...")
        gimbal.stop_all_servos()
        gimbal.reset_to_mid()
    finally:
        gimbal.disconnect()
