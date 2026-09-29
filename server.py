#!/usr/bin/env python
"""Compatibility entry point; the installed engine lives in the rag package."""
import sys
from rag import server

if __name__ == '__main__':
    server.main()
else:
    sys.modules[__name__] = server
