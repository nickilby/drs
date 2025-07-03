#!/usr/bin/env python3
"""
Test runner script for vCenter DRS Compliance Dashboard.

This script runs the test suite and provides a summary of results.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Main test runner function."""
    print("🧪 vCenter DRS Compliance Dashboard - Test Suite")
    print("=" * 60)
    
    # Change to the vcenter_drs directory
    os.chdir(Path(__file__).parent)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from vcenter_drs directory.")
        sys.exit(1)
    
    # Test results tracking
    results = []
    
    # 1. Check Python syntax
    results.append(run_command(
        ["python", "-m", "py_compile", "streamlit_app.py"],
        "Python syntax check - streamlit_app.py"
    ))
    
    results.append(run_command(
        ["python", "-m", "py_compile", "main.py"],
        "Python syntax check - main.py"
    ))
    
    # 2. Check import statements
    results.append(run_command(
        ["python", "-c", "import streamlit_app; print('✅ streamlit_app imports successfully')"],
        "Import check - streamlit_app"
    ))
    
    results.append(run_command(
        ["python", "-c", "import main; print('✅ main imports successfully')"],
        "Import check - main"
    ))
    
    # 3. Run linting
    results.append(run_command(
        ["python", "-m", "flake8", ".", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
        "Linting - Error checking"
    ))
    
    results.append(run_command(
        ["python", "-m", "flake8", ".", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=88", "--statistics"],
        "Linting - Style checking"
    ))
    
    # 4. Run type checking
    results.append(run_command(
        ["python", "-m", "mypy", ".", "--ignore-missing-imports"],
        "Type checking - mypy"
    ))
    
    # 5. Run unit tests
    results.append(run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Unit tests - pytest"
    ))
    
    # 6. Run tests with coverage
    results.append(run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=term-missing"],
        "Unit tests with coverage"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 