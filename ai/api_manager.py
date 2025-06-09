#!/usr/bin/env python
# api_manager.py - Helper functions for managing API calls in GitHub Actions environment
import os
import random
import time
import sys


def setup_environment():
    """Sets additional environment variables if running in GitHub Actions"""
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    use_optimal_rpm = os.environ.get("GEMINI_OPTIMAL_RPM") == "true"

    # 禁用 LangSmith 跟踪，避免身份验证错误
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGCHAIN_CALLBACKS"] = "none"

    if is_github_actions:
        print(
            "Running in GitHub Actions environment. Configuring for optimal API usage.",
            file=sys.stderr,
        )

        # Set longer timeouts for GitHub Actions environment
        os.environ["LANGCHAIN_GENAI_MAX_RETRIES"] = "5"
        os.environ["LANGCHAIN_GENAI_API_TIMEOUT"] = "120"

        # Slow down API requests in GitHub Actions to prevent quota issues
        os.environ["GITHUB_ACTIONS_API_DELAY"] = "true"

        return True
    elif use_optimal_rpm:
        print("Running with optimal RPM settings for Gemini API.", file=sys.stderr)
        os.environ["GITHUB_ACTIONS_API_DELAY"] = "true"
        return True
    else:
        print(
            "Running in local environment. Using standard API configuration.",
            file=sys.stderr,
        )
        return False


def smart_delay(min_seconds=2, max_seconds=5):
    """Adds a delay between API calls, optimized for Gemini API limits

    Gemini 2.0 Flash 限制:
    - RPM (每分钟请求数): 15
    - TPM (每分钟 token 数): 1,000,000
    - RPD (每天请求数): 1,000 (免费层级)

    按RPM计算，理论上每4秒可以发送1个请求（60/15=4）
    """
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    use_long_delay = os.environ.get("GITHUB_ACTIONS_API_DELAY") == "true"

    if is_github_actions or use_long_delay:
        # GitHub Actions环境中使用4-5秒的延迟（稍高于理论最小值）
        delay = 4.0 + random.random()
    else:
        # 本地开发环境使用较短延迟
        delay = min_seconds + random.random() * (max_seconds - min_seconds)

    print(f"Adding delay of {delay:.1f} seconds between API calls", file=sys.stderr)
    time.sleep(delay)
    return delay


if __name__ == "__main__":
    # Test the functions
    setup_environment()
    smart_delay(5, 10)
    print("API Manager tested successfully", file=sys.stderr)
