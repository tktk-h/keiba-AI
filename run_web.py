"""ローカルWebアプリを起動: http://localhost:5000 をブラウザで開く。"""
from web.app import app

if __name__ == "__main__":
    print("http://localhost:5000 をブラウザで開いてください(終了は Ctrl+C)")
    app.run(host="127.0.0.1", port=5000, debug=False)
