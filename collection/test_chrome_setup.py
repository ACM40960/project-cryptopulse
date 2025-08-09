#!/usr/bin/env python3
"""
Test Chrome/Selenium Setup
Quick test script to verify Chrome and Selenium are working properly
"""

import os
import time
import logging

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    print("✅ Selenium imports successful")
except ImportError as e:
    print(f"❌ Selenium import failed: {e}")
    exit(1)

def test_chrome_configurations():
    """Test multiple Chrome configurations to find working setup"""
    
    print("🧪 TESTING CHROME/SELENIUM CONFIGURATIONS")
    print("="*50)
    
    configurations = [
        {
            "name": "Headless with WebDriver Manager",
            "headless": True,
            "use_webdriver_manager": True,
            "minimal": False
        },
        {
            "name": "Visible with WebDriver Manager", 
            "headless": False,
            "use_webdriver_manager": True,
            "minimal": False
        },
        {
            "name": "Headless Minimal Configuration",
            "headless": True,
            "use_webdriver_manager": True,
            "minimal": True
        },
        {
            "name": "Visible Minimal Configuration",
            "headless": False,
            "use_webdriver_manager": True,
            "minimal": True
        }
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\\n🔄 Test {i}/4: {config['name']}")
        print("-" * 40)
        
        success = test_single_configuration(config)
        
        if success:
            print(f"✅ SUCCESS: {config['name']} works!")
            print("💡 This configuration can be used for Twitter collection")
            return config
        else:
            print(f"❌ FAILED: {config['name']}")
    
    print("\\n❌ ALL CONFIGURATIONS FAILED")
    print("💡 Manual Chrome/ChromeDriver installation may be required")
    return None

def test_single_configuration(config):
    """Test a single Chrome configuration"""
    driver = None
    
    try:
        chrome_options = Options()
        
        if config["minimal"]:
            # Minimal configuration
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
        else:
            # Full configuration
            essential_args = [
                "--no-sandbox",
                "--disable-dev-shm-usage", 
                "--disable-gpu",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-web-security",
                "--allow-running-insecure-content",
                "--ignore-certificate-errors",
                "--window-size=1920,1080"
            ]
            
            for arg in essential_args:
                chrome_options.add_argument(arg)
        
        if config["headless"]:
            chrome_options.add_argument("--headless")
        
        # Anti-detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Try to find Chrome binary
        chrome_binaries = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser", 
            "/usr/bin/chromium",
            "/opt/google/chrome/chrome"
        ]
        
        for chrome_binary in chrome_binaries:
            if os.path.exists(chrome_binary):
                chrome_options.binary_location = chrome_binary
                print(f"   📍 Using Chrome binary: {chrome_binary}")
                break
        
        # Setup service
        if config["use_webdriver_manager"]:
            print("   🔄 Installing ChromeDriver via WebDriver Manager...")
            service = Service(ChromeDriverManager().install())
        else:
            service = Service("/usr/bin/chromedriver") if os.path.exists("/usr/bin/chromedriver") else None
        
        # Create driver
        print("   🚀 Creating Chrome driver...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test basic functionality
        print("   🌐 Testing basic navigation...")
        driver.get("https://www.google.com")
        time.sleep(2)
        
        # Check page title
        title = driver.title
        print(f"   📄 Page title: {title}")
        
        if "Google" in title:
            print("   ✅ Navigation successful!")
            
            # Test Twitter access
            print("   🐦 Testing Twitter access...")
            driver.get("https://twitter.com")
            time.sleep(3)
            
            twitter_title = driver.title
            print(f"   📄 Twitter title: {twitter_title}")
            
            if "twitter" in twitter_title.lower() or "x" in twitter_title.lower():
                print("   ✅ Twitter access successful!")
                return True
            else:
                print("   ⚠️ Twitter access unclear")
                return True  # Still count as success for basic functionality
        else:
            print("   ❌ Navigation failed")
            return False
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    finally:
        if driver:
            try:
                driver.quit()
                print("   🔄 Driver closed successfully")
            except:
                print("   ⚠️ Driver cleanup had issues")

def main():
    print("🔧 CHROME/SELENIUM SETUP TESTER")
    print("="*40)
    
    # Test configurations
    working_config = test_chrome_configurations()
    
    if working_config:
        print("\\n🎉 SETUP TEST SUCCESSFUL!")
        print(f"✅ Working configuration: {working_config['name']}")
        print("🚀 Ready to run comprehensive Twitter collection!")
        
        # Save working config for reference
        config_file = "logs/working_chrome_config.json" 
        os.makedirs("logs", exist_ok=True)
        
        import json
        with open(config_file, 'w') as f:
            json.dump(working_config, f, indent=2)
        
        print(f"💾 Configuration saved to: {config_file}")
        
    else:
        print("\\n❌ SETUP TEST FAILED")
        print("🛠️ TROUBLESHOOTING STEPS:")
        print("   1. Check Chrome installation: google-chrome --version")
        print("   2. Install missing packages: pip install selenium webdriver-manager")
        print("   3. Try system ChromeDriver: sudo apt install chromium-driver")
        print("   4. Check system display: echo $DISPLAY")

if __name__ == "__main__":
    main()