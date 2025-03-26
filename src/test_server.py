#!/usr/bin/env python
"""
Server Test Script
=================
This script tests if the baby monitor Flask server is running correctly and responding to API requests.
"""

import requests
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server URL
SERVER_URL = "http://localhost:5000"

# API endpoints to test
ENDPOINTS = {
    "pages": [
        "/",                      # Main page
        "/metrics",               # Metrics page
        "/repair",                # Repair tools page
    ],
    "cameras": [
        "/api/cameras",           # Get cameras
    ],
    "emotion": [
        "/api/emotion/models",    # Get emotion models
    ],
    "system": [
        "/api/system/check",      # Check system status
        "/api/system/info",       # Get system info
    ]
}

def test_endpoint(url):
    """Test if an endpoint responds correctly"""
    full_url = f"{SERVER_URL}{url}"
    try:
        logger.info(f"Testing endpoint: {url}")
        response = requests.get(full_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ {url} - Success ({response.status_code})")
            return True
        else:
            logger.error(f"❌ {url} - Failed with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ {url} - Error: {str(e)}")
        return False

def main():
    """Main function to test the server"""
    # First check if the server is running
    try:
        logger.info("Checking if the server is running...")
        response = requests.get(f"{SERVER_URL}/", timeout=2)
        logger.info(f"Server is running! Status code: {response.status_code}")
    except requests.exceptions.RequestException:
        logger.error("❌ Server is not running or not accessible at localhost:5000")
        logger.info("Make sure to run 'python src/run_server.py' first")
        return 1
    
    # Test all endpoints
    success_count = 0
    total_count = 0
    
    for category, endpoints in ENDPOINTS.items():
        logger.info(f"\nTesting {category.upper()} endpoints:")
        for endpoint in endpoints:
            if test_endpoint(endpoint):
                success_count += 1
            total_count += 1
            time.sleep(0.5)  # Small delay to avoid overwhelming the server
    
    # Print summary
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"\n==== Test Summary ====")
    logger.info(f"Endpoints tested: {total_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_count - success_count}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if success_count == total_count:
        logger.info("✅ All tests passed! Server is working correctly.")
        return 0
    else:
        logger.warning("⚠️ Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 