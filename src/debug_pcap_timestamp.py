from pathlib import Path

from scapy.utils import RawPcapReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCAP = PROJECT_ROOT / "data" / "raw" / "Monday-WorkingHours.pcap"


print("=" * 80)
print("PCAP TIMESTAMP DEBUG")
print("=" * 80)

print(f"PCAP: {PCAP}")

with RawPcapReader(str(PCAP)) as reader:

    pkt_data, metadata = next(reader)

print("\nMetadata type:")
print(type(metadata))

print("\nMetadata repr:")
print(repr(metadata))

print("\nMetadata fields:")
print(metadata._fields)

print("\nMetadata values:")

for field in metadata._fields:
    print(
        f"  {field:20s} = "
        f"{getattr(metadata, field)!r}"
    )

print("\nRaw packet length:")
print(len(pkt_data))

print("\nSpecific timestamp fields:")

print("tshigh :", metadata.tshigh)
print("tslow  :", metadata.tslow)
print("tsresol:", metadata.tsresol)

print("\nCombined raw timestamp:")

raw_timestamp = (
    (int(metadata.tshigh) << 32)
    | int(metadata.tslow)
)

print(raw_timestamp)