"""Test-wide settings."""

import os

# Tuning results aren't read from or written to the user's cache: each test searches afresh (the cache's
# own tests turn it on, in a temporary directory).
os.environ["AUTOKERNEL_CACHE"] = "0"
