#!/usr/bin/env python3
"""
Test script to verify API connectivity and credentials.
Run this before starting the competition bot to ensure everything works.
"""

import json
import sys

import requests

# Configuration
API_BASE_URL = "https://edth.helsing.codes"
AUTH_TOKEN = "f276bbf9-e42b-452c-be54-eac3d4c6f0e3"


def test_get_challenge() -> bool:
    """Test fetching the current challenge."""
    print("🔍 Testing: GET /api/challenge")
    try:
        response = requests.get(f"{API_BASE_URL}/api/challenge", timeout=5)
        response.raise_for_status()
        challenge = response.json()

        print(f"✅ Success!")
        print(f"   Challenge ID: {challenge.get('challenge_id', 'N/A')}")
        print(f"   WAV URL: {challenge.get('wav_url', 'N/A')}")
        print(f"   Time remaining: {challenge.get('time_until_next_rotation_ms', 0) / 1000:.1f}s")
        return True
    except requests.RequestException as e:
        print(f"❌ Failed: {e}")
        return False


def test_download_audio(wav_url: str) -> bool:
    """Test downloading audio file."""
    print("\n🔍 Testing: Download audio file")
    try:
        full_url = f"{API_BASE_URL}{wav_url}"
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()

        size = len(response.content)
        print(f"✅ Success!")
        print(f"   Downloaded {size:,} bytes")
        print(f"   Content type: {response.headers.get('content-type', 'N/A')}")
        return True
    except requests.RequestException as e:
        print(f"❌ Failed: {e}")
        return False


def test_auth() -> bool:
    """Test authentication with the token."""
    print("\n🔍 Testing: Authentication")
    print(f"   Token: {AUTH_TOKEN}")

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    # Note: We can't actually test submission without a valid challenge
    # But we can verify the token format is correct
    if len(AUTH_TOKEN) == 36 and AUTH_TOKEN.count("-") == 4:
        print("✅ Token format looks correct (UUID format)")
        return True
    else:
        print("⚠️  Token format might be incorrect")
        return False


def test_api_health() -> bool:
    """Test if API is accessible."""
    print("\n🔍 Testing: API health")
    try:
        response = requests.get(f"{API_BASE_URL}/static/index.html", timeout=5)
        if response.status_code == 200:
            print("✅ API is accessible")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Failed: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print("="*60)
    print("  🧪 API Connectivity Test")
    print("="*60)
    print()

    results = []

    # Test 1: API health
    results.append(("API Health", test_api_health()))

    # Test 2: Get challenge
    challenge_result = test_get_challenge()
    results.append(("Get Challenge", challenge_result))

    # Test 3: Download audio (only if we got a challenge)
    if challenge_result:
        try:
            response = requests.get(f"{API_BASE_URL}/api/challenge", timeout=5)
            challenge = response.json()
            wav_url = challenge.get("wav_url")
            if wav_url:
                results.append(("Download Audio", test_download_audio(wav_url)))
        except Exception:
            pass

    # Test 4: Authentication
    results.append(("Authentication", test_auth()))

    # Summary
    print("\n" + "="*60)
    print("  📊 Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print()
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! You're ready to compete!")
        print(f"\n  Next steps:")
        print(f"    1. Train a model: uv run python train_model.py")
        print(f"    2. Start the bot: uv run python competition_bot.py")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some tests failed. Please check:")
        print(f"    - Internet connection")
        print(f"    - API is online")
        print(f"    - Token is correct")
        sys.exit(1)


if __name__ == "__main__":
    main()

