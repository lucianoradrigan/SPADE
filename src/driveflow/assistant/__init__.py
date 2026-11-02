"""SPADE assistant (round Assistant, phase 1): a conversational helper that configures simulations.

It runs on a LOCAL language model served by Ollama (decision B1, 2026-11-02); nothing leaves the
machine and no Python dependency is added (B2). The language model never writes a number into the
draft: every value enters through a tool and carries its origin (``draft.ORIGINS``), and
``numbers`` decides whether a number was really given by the user or by a tool result.

Planning documents: docs/assistant/ (Spanish).
"""
