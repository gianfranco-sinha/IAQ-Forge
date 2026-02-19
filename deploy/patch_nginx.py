#!/usr/bin/env python3
"""Inject iaq4j location block into enviro-sensors.uk nginx config."""
import sys

CONF = "/etc/nginx/sites-enabled/enviro-sensors.uk"
OUT = "/tmp/enviro-sensors.uk.new"

SNIPPET = """\
    # ----------------------------
    # iaq4j API
    # ----------------------------
    location /iaq/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 120s;
    }

"""

MARKER = "    # ----------------------------\n    # Proxy all requests to Grafana"

with open(CONF) as f:
    content = f.read()

if "location /iaq/" in content:
    print("iaq4j location block already present")
    sys.exit(0)

if MARKER not in content:
    print("ERROR: could not find marker in config")
    sys.exit(1)

content = content.replace(MARKER, SNIPPET + MARKER)
with open(OUT, "w") as f:
    f.write(content)
print(f"Patched config written to {OUT}")
