#!/usr/bin/env python3
import datetime
import builtins

# Store the original print function
original_print = builtins.print

# Define a new print function that adds timestamps
def timestamped_print(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]')
    original_print(timestamp, *args, **kwargs)

# Replace the built-in print function with our timestamped version
builtins.print = timestamped_print

