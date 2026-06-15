"""ローカルWebアプリを起動し、ブラウザを自動で開く。終了は Ctrl+C。"""
import threading
import webbrowser
from web.app import app

URL = "http://localhost:5000"

if __name__ == "__main__":
    print(f"{URL} を自動で開きます。終了するにはこのウィンドウで Ctrl+C。")
    # サーバー起動直後にブラウザを開く(1.5秒待ってから)
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
