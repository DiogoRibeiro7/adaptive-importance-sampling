"""
Configuration for architecture diagram generation
"""

# Modules to exclude from diagrams
EXCLUDED_MODULES = ["migrations", "tests", "__pycache__"]

# Classes to highlight in diagrams
IMPORTANT_CLASSES = ["User", "Order", "Product"]

# Custom module groupings
MODULE_GROUPS = {
    "api": ["routes", "controllers", "endpoints"],
    "core": ["models", "schemas", "services"],
    "utils": ["helpers", "validators", "decorators"],
}
