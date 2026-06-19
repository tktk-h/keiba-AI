"""ローカルWebアプリを起動し、ブラウザを自動で開く。終了は Ctrl+C。

環境変数で公開範囲を変えられる(既定はローカルのみ):
  KEIBA_HOST=0.0.0.0  … LAN/トンネル公開(スマホから http://<PCのIP>:ポート)
  KEIBA_PORT=5000     … ポート変更
"""
import os
import threading
import webbrowser
from web.app import app

HOST = os.environ.get("KEIBA_HOST", "127.0.0.1")
PORT = int(os.environ.get("KEIBA_PORT", "5000"))
URL = f"http://localhost:{PORT}"

if __name__ == "__main__":
    print(f"{URL} を自動で開きます(HOST={HOST})。終了するにはこのウィンドウで Ctrl+C。")
    # サーバー起動直後にブラウザを開く(1.5秒待ってから)
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    app.run(host=HOST, port=PORT, debug=False)
