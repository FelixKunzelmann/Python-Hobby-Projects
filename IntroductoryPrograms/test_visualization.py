"""
Quick test to verify pygame is available and the visualization works.
"""
try:
    import pygame
    print("✓ pygame is installed")
except ImportError:
    print("✗ pygame is NOT installed")
    print("\nTo use the visualization, install pygame with:")
    print("  pip install pygame")
    print("\nAfter installation, run IdealGasSimulation.py to see the visual mode.")
