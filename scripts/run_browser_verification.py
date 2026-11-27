"""
Harness to start local HTTP server and Chrome headless, run the browser verification agent, and cleanly clean up.
"""

import subprocess
import time
import os
import sys
import socketserver
import http.server
import functools
import threading

PORT = 8000
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def run_harness():
    print("Starting background HTTP server on port", PORT)
    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="web")
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    print(f"Launching Chrome Headless with CDP on port 9222...")
    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        "about:blank"
    ]
    chrome_proc = subprocess.Popen(chrome_cmd)
    time.sleep(2)

    try:
        print("Executing Node.js Browser Verifier (scripts/browser_verifier.mjs)...")
        node_res = subprocess.run(["node", "scripts/browser_verifier.mjs"], capture_output=True, text=True, encoding="utf-8")
        print(node_res.stdout)
        if node_res.stderr:
            print("STDERR:\n", node_res.stderr)
        if node_res.returncode != 0:
            print(f"[FAIL] Node verifier exited with code {node_res.returncode}")
            sys.exit(node_res.returncode)
    finally:
        print("Shutting down Chrome and HTTP server...")
        chrome_proc.terminate()
        httpd.shutdown()
        print("Cleanup completed.")

if __name__ == "__main__":
    run_harness()
