from scapy.all import *
def handler(packet):
    print(packet.summary())
sniff(iface="lo", prn=handler, store=0)


#sudo python3 sniffer_A.py