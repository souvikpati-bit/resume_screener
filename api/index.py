"""Vercel entry point: every URL is rewritten here (vercel.json) and served by the same handler as app.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import Handler  # noqa: E402


class handler(Handler):
    pass
