# Security

## Reporting a problem

Please report a security problem privately, through GitHub's private vulnerability reporting: open [github.com/IamCooller/larder](https://github.com/IamCooller/larder), go to the Security tab and choose "Report a vulnerability". Do not open a public issue for it. Say what you ran, what happened and what you expected. This is a one-person project, so a reply can take a few days.

## What listens, and where

**The server.** `server.py` binds 127.0.0.1 by default, so only programs on this computer can reach it. It refuses a request whose Origin or Host header is not this computer, so a web page you visit cannot use it.

Binding it to another address (`--host`) needs `--api-key`: without one the server refuses to start. With a key, every API request must carry it, as `Authorization: Bearer <key>` or `x-api-key: <key>`, and the key check takes the place of the Origin and Host check. The chat page and its files load without the key, so the page can ask for it. Without `--tls-cert` and `--tls-key` the key and every request cross the network in clear text.

**`./larder share` (experimental).** It lets the menu on another of your computers use this computer's model over the LAN.

- It listens on the LAN and sends a UDP broadcast beacon, so menus on the same network can find it.
- It needs `openssl`. The first start makes a self-signed certificate; the control service, the pairing and the model server then speak TLS, and the other computer pins the certificate at pairing, so a connection with a different certificate is refused. Without `openssl`, share refuses to start unless you pass `--no-tls`, which sends the pairing and every request in clear text.
- Pairing uses a 12-character code shown on the shared computer and a certificate fingerprint to compare on both screens. Both sides prove they know the code (PBKDF2 and an HMAC challenge in both directions); the shared computer then issues a long token.
- The tokens are kept in files only your user can read (mode 0600) under `~/.cache/moe-ssd-offload/`: `shared-clients.json` on the shared computer, `paired.json` on the other one. Every later call is signed with the token and accepted once, within 60 seconds.
- To revoke a pairing, delete its token from `shared-clients.json` and restart `./larder share`, or delete the device's row from `paired.json`. Either side alone ends it.
- The menu reaches a shared model through a local proxy on 127.0.0.1, which refuses foreign Origin and Host headers like the server does.

**Devices.** The menu uses your own ssh setup and keys.

- The scan of your LAN (a TCP attempt to port 22 of each address on the local /24, then the ARP table) runs only after you confirm it.
- Hosts from `~/.ssh/config` and Tailscale are checked over ssh when you turn devices on (`./larder --devices`). With devices off, they are also checked after you confirm the scan. A host that only the scan found gets no ssh until you open it.
- A computer found by its share beacon is never reached over ssh. The menu talks only to its share service, with the pairing and signed calls above.
- Opening a device lists, starts, deletes and updates models there over ssh. A model started there is reached through an ssh port forward to its 127.0.0.1.
- `u` updates the engine there: it runs `git pull` in that checkout, or, when that does not bring this computer's commit, sends this checkout as a git bundle over ssh.
- Setting up a device runs the installer there over ssh, from the public installer or with this checkout sent as a git bundle.

**Everything else.** The engine has no telemetry and no update check. The README's "Privacy and network" section lists every network use, and its "Uninstall" section lists every folder the engine writes.
