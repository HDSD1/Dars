"""Reference binary messages for the R1.3 signaling experiment.

This is an application-layer serialization benchmark, not a radio protocol.
All floating-point fields use network-endian IEEE-754 binary64, preserving
the simulator's physical-state precision. No native struct padding is used.

Message layout:
    header: magic[4], version:u8, kind:u8, episode:u32, slot:u16,
            sender:u16, record_count:u16                              (16 B)
    user record: id:u16, x:f64, y:f64, task_bits:f64, active:u8        (27 B)
    peer own state: x:f64, y:f64, battery_j:f64                        (24 B)
    grant: id:u16, serving_uav:i16, offload:f64, bandwidth_hz:f64,
           bandstart_hz:f64, status:u8                               (29 B)

Kinds are 1=user report, 2=peer exchange, 3=user grant. A grant's status is
0=execute or 1=abort. An abort retains its status and fixed length; it must
not be interpreted as an executed all-local decision.
"""

import math
import struct


MAGIC = b"DR13"
VERSION = 1
USER_REPORT = 1
PEER_EXCHANGE = 2
USER_GRANT = 3
EXECUTE = 0
ABORT = 1

HEADER = struct.Struct("!4sBBIHHH")
USER_RECORD = struct.Struct("!HdddB")
OWN_STATE = struct.Struct("!ddd")
GRANT_RECORD = struct.Struct("!HhdddB")
UDP_IPV4_HEADER_BYTES = 28  # UDP 8 B + IPv4 without options 20 B.


def pack_header(kind, episode, slot, sender, count):
    """Return the fixed 16-byte header; identifiers are unsigned integers."""
    if kind not in (USER_REPORT, PEER_EXCHANGE, USER_GRANT):
        raise ValueError(f"Unknown message kind: {kind}")
    return HEADER.pack(MAGIC, VERSION, kind, episode, slot, sender, count)


def unpack_header(data):
    """Decode a header from a complete message or a 16-byte header."""
    if len(data) < HEADER.size:
        raise ValueError("Truncated message header")
    magic, version, kind, episode, slot, sender, count = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported R1.3 message format")
    if kind not in (USER_REPORT, PEER_EXCHANGE, USER_GRANT):
        raise ValueError(f"Unknown message kind: {kind}")
    return dict(version=version, kind=kind, episode=episode, slot=slot,
                sender=sender, count=count)


def _pack_user(record):
    active = int(record["active"])
    if active not in (0, 1):
        raise ValueError("The task activation flag must be 0 or 1")
    return USER_RECORD.pack(record["id"], record["position"][0],
                            record["position"][1], record["task_bits"], active)


def _unpack_user(data, offset):
    uid, x, y, task_bits, active = USER_RECORD.unpack_from(data, offset)
    if active not in (0, 1):
        raise ValueError("The task activation flag must be 0 or 1")
    return dict(id=uid, position=[x, y], task_bits=task_bits,
                active=bool(active))


def encode_user_report(record, *, episode, slot, sender=None):
    """Encode one user record; the sender defaults to that user's ID."""
    if sender is None:
        sender = record["id"]
    return pack_header(USER_REPORT, episode, slot, sender, 1) + _pack_user(record)


def decode_user_report(data):
    """Return {'header': ..., 'record': <user record dictionary>}."""
    header = unpack_header(data)
    if (header["kind"] != USER_REPORT or header["count"] != 1
            or len(data) != HEADER.size + USER_RECORD.size):
        raise ValueError("Invalid user-report layout")
    return dict(header=header, record=_unpack_user(data, HEADER.size))


def encode_peer(own_state, records, *, episode, slot, sender):
    """Encode one UAV's physical state followed by its collected UE records.

    own_state is {'position': [x, y], 'battery_j': float}.
    records is an iterable of user-record dictionaries. Its order is preserved.
    """
    records = list(records)
    body = OWN_STATE.pack(own_state["position"][0], own_state["position"][1],
                          own_state["battery_j"])
    return (pack_header(PEER_EXCHANGE, episode, slot, sender, len(records))
            + body + b"".join(_pack_user(record) for record in records))


def decode_peer(data):
    """Return {'header': ..., 'own_state': ..., 'records': [...]}."""
    header = unpack_header(data)
    expected = HEADER.size + OWN_STATE.size + header["count"] * USER_RECORD.size
    if header["kind"] != PEER_EXCHANGE or len(data) != expected:
        raise ValueError("Invalid peer-exchange layout")
    x, y, battery = OWN_STATE.unpack_from(data, HEADER.size)
    start = HEADER.size + OWN_STATE.size
    records = [_unpack_user(data, start + i * USER_RECORD.size)
               for i in range(header["count"])]
    return dict(header=header, own_state=dict(position=[x, y], battery_j=battery),
                records=records)


def encode_grant(grant, *, episode, slot, sender):
    """Encode a user's final authorization, including an explicit abort flag.

    grant keys: id, serving_uav (0/1 or -1 for local), offload,
    bandwidth_hz, bandstart_hz, status (EXECUTE=0 or ABORT=1).
    The serving UAV is distinct from sender, the reporting UAV delivering it.
    """
    if grant["status"] not in (EXECUTE, ABORT):
        raise ValueError("Grant status must be EXECUTE=0 or ABORT=1")
    body = GRANT_RECORD.pack(grant["id"], grant["serving_uav"], grant["offload"],
                            grant["bandwidth_hz"], grant["bandstart_hz"],
                            grant["status"])
    return pack_header(USER_GRANT, episode, slot, sender, 1) + body


def decode_grant(data):
    """Return {'header': ..., 'grant': <grant dictionary>}."""
    header = unpack_header(data)
    if (header["kind"] != USER_GRANT or header["count"] != 1
            or len(data) != HEADER.size + GRANT_RECORD.size):
        raise ValueError("Invalid user-grant layout")
    uid, serving, offload, bandwidth, start, status = GRANT_RECORD.unpack_from(
        data, HEADER.size)
    if status not in (EXECUTE, ABORT):
        raise ValueError("Grant status must be EXECUTE=0 or ABORT=1")
    grant = dict(id=uid, serving_uav=serving, offload=offload,
                 bandwidth_hz=bandwidth, bandstart_hz=start, status=status)
    return dict(header=header, grant=grant)


def analytic_counts(user_count, uav_count=2):
    """Return packet and byte counts for the specified two-UAV protocol.

    All users report once and receive one grant (including abort grants).
    Each user report is forwarded once between the two UAVs. The UDP/IPv4
    total assumes one message per datagram, no IPv4 options or fragmentation.
    """
    if uav_count != 2:
        raise ValueError("This protocol's exchange accounting is for two UAVs")
    up = user_count * (HEADER.size + USER_RECORD.size)
    peer = uav_count * (HEADER.size + OWN_STATE.size) + user_count * USER_RECORD.size
    down = user_count * (HEADER.size + GRANT_RECORD.size)
    packet_count = 2 * user_count + uav_count
    app = up + peer + down
    return dict(user_up_bytes=up, peer_bytes=peer, user_down_bytes=down,
                message_count=packet_count, application_bytes=app,
                udp_ipv4_bytes=app + packet_count * UDP_IPV4_HEADER_BYTES)


def self_test():
    """Check binary64 round trips, message identity and analytic packet counts."""
    assert (HEADER.size, USER_RECORD.size, OWN_STATE.size, GRANT_RECORD.size) == (
        16, 27, 24, 29)
    users = [dict(id=i, position=[math.nextafter(0.1 + i, math.inf), -0.25 * i],
                  task_bits=(2_500_000.125 + i if i % 2 else 0.0),
                  active=bool(i % 2)) for i in range(30)]
    own = [dict(position=[12.25, 50.125], battery_j=math.nextafter(40000., 0.)),
           dict(position=[250.125, 399.75], battery_j=20000.5)]
    kwargs = dict(episode=700001, slot=49)
    packets = []
    for user in users:
        msg = encode_user_report(user, **kwargs)
        decoded = decode_user_report(msg)
        assert decoded["record"] == user
        assert decoded["header"]["sender"] == user["id"]
        assert decoded["header"]["episode"] == kwargs["episode"]
        assert decoded["header"]["slot"] == kwargs["slot"]
        assert encode_user_report(decoded["record"], **kwargs) == msg
        packets.append(msg)
    for k, records in enumerate((users[:13], users[13:])):
        msg = encode_peer(own[k], records, sender=k, **kwargs)
        decoded = decode_peer(msg)
        assert decoded["own_state"] == own[k]
        assert decoded["records"] == records
        assert encode_peer(decoded["own_state"], decoded["records"],
                           sender=k, **kwargs) == msg
        packets.append(msg)
    accepted = []
    aborted = []
    for user in users:
        grant = dict(id=user["id"], serving_uav=user["id"] % 2,
                     offload=math.nextafter(0.5, 1.), bandwidth_hz=33333.125,
                     bandstart_hz=100000.5, status=EXECUTE)
        msg = encode_grant(grant, sender=0, **kwargs)
        decoded = decode_grant(msg)
        assert decoded["grant"] == grant
        assert encode_grant(decoded["grant"], sender=0, **kwargs) == msg
        accepted.append(msg)
        abort = dict(id=user["id"], serving_uav=-1, offload=0.,
                     bandwidth_hz=0., bandstart_hz=0., status=ABORT)
        abort_msg = encode_grant(abort, sender=0, **kwargs)
        assert decode_grant(abort_msg)["grant"] == abort
        assert len(abort_msg) == len(msg)
        aborted.append(abort_msg)
    counts = analytic_counts(30)
    assert counts["application_bytes"] == 3530
    assert counts["udp_ipv4_bytes"] == 5266
    for grants in (accepted, aborted):
        all_packets = packets + grants
        assert len(all_packets) == counts["message_count"] == 62
        assert sum(map(len, all_packets)) == counts["application_bytes"]
        assert sum(len(packet) + UDP_IPV4_HEADER_BYTES for packet in all_packets) == (
            counts["udp_ipv4_bytes"])
    return dict(status="passed", header_bytes=HEADER.size,
                user_record_bytes=USER_RECORD.size,
                own_state_bytes=OWN_STATE.size,
                grant_record_bytes=GRANT_RECORD.size, **counts)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
