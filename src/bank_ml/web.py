"""Flask serves the interface; same-origin JavaScript uses the existing API."""
from flask import Flask, render_template


def create_web_app():
    web = Flask(__name__)
    web.config["MAX_CONTENT_LENGTH"] = 32 * 1024

    @web.get("/")
    def home():
        return render_template("index.html")

    @web.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    return web
