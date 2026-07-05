from pymodbus.client import ModbusTcpClient
c = ModbusTcpClient("192.168.168.1", port=502, timeout=3)
print("connect:", c.connect())
rr = c.read_holding_registers(1, count=10, device_id=1)   # device_id per your pymodbus 3.9+
print("error:", rr.isError() if rr else "None")
print(rr.registers if rr and not rr.isError() else "no data")
c.close()