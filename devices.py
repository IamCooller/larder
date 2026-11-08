#!/usr/bin/env python3
"""The other computers on this network for the launcher menu's Devices tab, found by trying ssh's
port on this computer's /24, reading the ARP table and listing the peers `tailscale status` knows,
with nothing installed or sent."""
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import share

# ssh is the one service worth a probe here: it says "you can log in and run something".
PORTS = (22,)
TIMEOUT = 0.4


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def local_address():
    """This computer's address on the network it reaches the internet through, or None, read from a
    UDP socket that is connected but sends nothing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))              # a documentation address: never answers
            return s.getsockname()[0]
        except OSError:
            return None


def subnet(me=None):
    """This computer's /24 as "a.b.c.0/24", the addresses a scan would try, or None off a network."""
    me = me or local_address()
    if not me or me.startswith("127."):
        return None
    return me.rsplit(".", 1)[0] + ".0/24"


def beacons(timeout=1.5, port=None):
    """The share beacons heard in the last `timeout` seconds, one row per shared device with its
    name, address and control port."""
    port = port or share.BEACON_PORT
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", port))
    except OSError:
        return []
    s.settimeout(0.5)
    out, seen, end = [], set(), time.time() + timeout
    while time.time() < end:
        try:
            data, addr = s.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            msg = json.loads(data)
        except ValueError:
            continue
        if msg.get("magic") != share.BEACON_MAGIC:
            continue
        key = (msg.get("name") or addr[0], msg.get("port") or 0)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": msg.get("name") or addr[0], "host": addr[0],
                    "port": msg.get("port") or 0, "https": bool(msg.get("https"))})
    s.close()
    return out


def parse_arp(text):
    """[(ip, mac, name or "")] from `arp -an` (macOS, Linux) or `ip neigh` output, leaving out
    incomplete entries and multicast or broadcast rows."""
    out = []
    for line in text.splitlines():
        m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-fA-F:]{11,17})", line)    # macOS, Linux arp -a
        if m:
            name = line.split(" (")[0].strip()
            ip, mac = m[1], m[2]
        else:
            m = re.match(r"^(\d+\.\d+\.\d+\.\d+)\s+dev\s+\S+\s+lladdr\s+([0-9a-fA-F:]{11,17})", line)   # ip neigh
            if not m:
                continue
            ip, mac, name = m[1], m[2], ""
        if name == "?":
            name = ""
        first = int(ip.split(".")[0])
        if mac.lower() in ("ff:ff:ff:ff:ff:ff",) or 224 <= first <= 239 or ip.endswith(".255"):
            continue
        out.append((ip, ":".join(x.zfill(2) for x in mac.lower().split(":")), name))   # macOS drops leading zeros
    return out


def gateway(run=_run):
    """The default gateway's address (the router), or None."""
    if sys.platform == "darwin":
        m = re.search(r"gateway:\s*(\S+)", run(["route", "-n", "get", "default"]))
    else:
        m = re.search(r"default via (\S+)", run(["ip", "route", "show", "default"]))
    return m[1] if m else None


def resolve(ip, wait=1.5):
    """The name the resolver (mDNS included) gives an address, or "" when it takes too long."""
    box = []

    def look():
        try:
            box.append(socket.gethostbyaddr(ip)[0])
        except OSError:
            box.append("")
    t = threading.Thread(target=look, daemon=True)
    t.start()
    t.join(wait)
    return box[0] if box else ""


def probe(ip, ports=PORTS, timeout=TIMEOUT, connect=socket.create_connection):
    """(alive, open ports) for one address, where a refused connection still proves a computer
    answered."""
    alive, opened = False, []
    for p in ports:
        try:
            connect((ip, p), timeout).close()
            alive = True
            opened.append(p)
        except ConnectionRefusedError:
            alive = True
        except OSError:
            pass
    return alive, opened


def tailscale_peers(run=_run):
    """[{name, ip, os, online}] for the peers `tailscale status` lists; [] without the program."""
    exe = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    try:
        st = json.loads(run([exe, "status", "--json"]) or "{}")
    except ValueError:
        return []
    out = []
    for p in (st.get("Peer") or {}).values():
        ips = p.get("TailscaleIPs") or []
        out.append({"name": p.get("HostName") or p.get("DNSName", "").split(".")[0], "ip": ips[0] if ips else "",
                    "os": p.get("OS", ""), "online": bool(p.get("Online"))})
    return sorted(out, key=lambda d: (not d["online"], d["name"].lower()))


def scan(progress=None, run=_run, connect=socket.create_connection, me=None):
    """Look at the network once, only when the user asked for it, and return {"me", "subnet", "devices",
    "tailscale", "error"}, with each device {"ip", "name", "mac", "ports", "note"} sorted by address and
    `progress(done, total)` called as addresses are probed."""
    me = me or local_address()
    result = {"me": me, "subnet": subnet(me), "devices": [], "tailscale": [], "error": None}
    result["tailscale"] = tailscale_peers(run)
    if not result["subnet"]:
        result["error"] = "not connected to a network"
        return result
    base = me.rsplit(".", 1)[0]
    addrs = [f"{base}.{i}" for i in range(1, 255) if f"{base}.{i}" != me]
    seen = {}
    done = [0]
    lock = threading.Lock()

    def one(ip):
        alive, opened = probe(ip, connect=connect)
        with lock:
            done[0] += 1
            if progress:
                progress(done[0], len(addrs))
        if alive:
            seen[ip] = opened
    with ThreadPoolExecutor(max_workers=96) as pool:
        list(pool.map(one, addrs))
    # The attempts above made the system ask who has each address: hosts that drop
    # every connection (a firewall) still answered ARP.
    arp = {ip: (mac, name) for ip, mac, name in parse_arp(run(["arp", "-an"]) or run(["ip", "neigh"]))
           if ip.startswith(base + ".") and ip != me}
    gw = gateway(run)
    found = sorted(set(seen) | set(arp))
    with ThreadPoolExecutor(max_workers=16) as pool:           # a slow resolver costs one wait, not one per device
        names = dict(zip(found, pool.map(lambda ip: arp.get(ip, ("", ""))[1] or resolve(ip), found)))
    for ip in found:
        ports = seen.get(ip, [])
        note = ", ".join(filter(None, ["router" if ip == gw else "", "ssh" if 22 in ports else ""]))
        result["devices"].append({"ip": ip, "name": names[ip], "mac": arp.get(ip, ("", ""))[0], "ports": ports,
                                  "note": note})
    result["devices"].sort(key=lambda d: tuple(int(x) for x in d["ip"].split(".")))
    return result


if __name__ == "__main__":
    r = scan()
    print(f"this computer {r['me']}   network {r['subnet']}   {r['error'] or ''}")
    for d in r["devices"]:
        print(f"  {d['ip']:<15} {d['name'] or '-':<32} {d['mac'] or '-':<17} {d['note']}")
    for p in r["tailscale"]:
        print(f"  tailscale {p['name']:<24} {p['ip']:<15} {p['os']:<8} {'online' if p['online'] else 'offline'}")
