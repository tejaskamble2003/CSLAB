from scapy.all import *

A = "192.168.5.133" # spoofed source IP address
B = "192.168.5.135" # destination IP address
C = RandShort() # source port
D = 80 # destination port
payload = "Hello Hello Hello" # packet payload

while True:
    spoofed_packet = IP(src=A, dst=B) / TCP(sport=C, dport=D) / payload
    send(spoofed_packet)
    time.sleep(1)

#sudo python3 ip_spoofing.py -v -l lo
#sudo wireshark 
#sudo apt install wireshark