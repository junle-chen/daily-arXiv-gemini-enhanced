#!/usr/bin/env python
# Test script to verify the tenacity retry mechanism
# You can run this to test if the retry functionality works without making actual API calls

import sys
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Simulate the API exceptions
class ResourceExhausted(Exception):
    pass


class InternalServerError(Exception):
    pass


# Counter to track retry attempts
attempts = 0


@retry(
    reraise=True,
    stop=stop_after_attempt(4),  # Maximum 4 attempts
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff
    retry=retry_if_exception_type((ResourceExhausted, InternalServerError)),
)
def test_function(fail_times=3):
    """Test function that simulates failures a few times before succeeding"""
    global attempts
    attempts += 1

    print(f"Attempt #{attempts}", file=sys.stderr)

    if attempts <= fail_times:
        print(f"Simulating a rate limit error (ResourceExhausted)", file=sys.stderr)
        raise ResourceExhausted("429 You exceeded your current quota")

    print("Success on attempt #" + str(attempts), file=sys.stderr)
    return "Success!"


def main():
    global attempts

    print("Testing retry mechanism with tenacity...", file=sys.stderr)
    print("This will simulate 3 failures before succeeding", file=sys.stderr)

    try:
        result = test_function()
        print(f"\nFinal result: {result}", file=sys.stderr)
        print(f"Total attempts: {attempts}", file=sys.stderr)
    except Exception as e:
        print(f"\nFailed after {attempts} attempts: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
