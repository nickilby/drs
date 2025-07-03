"""
Main entry point for vCenter DRS application.

This module provides the command-line interface for the vCenter DRS system.
It handles data collection, connectivity testing, and serves as the main
execution point for the application.

Usage:
    python main.py                    # Collect data from vCenter
    python main.py check             # Test vCenter connectivity
    python main.py --help            # Show help information
"""

import sys
import time
import argparse
from typing import Optional

from vcenter_drs.api.vcenter_client_pyvomi import VCenterPyVmomiClient
from vcenter_drs.api.collect_and_store_metrics import main as collect_and_store_metrics_main
from logging_config import setup_logging, get_logger
from config import config
from exceptions import VCenterConnectionError, ConfigurationError


def check_connectivity() -> bool:
    """
    Test connectivity to vCenter Server.
    
    Returns:
        bool: True if connection successful, False otherwise
        
    Raises:
        VCenterConnectionError: If connection fails
    """
    logger = get_logger(__name__)
    
    try:
        # Validate configuration first
        if not config.validate():
            logger.error("Configuration validation failed")
            return False
        
        # Test connection
        with VCenterPyVmomiClient() as client:
            si = client.connect()
            logger.info("Successfully connected to vCenter")
            logger.info(f"vCenter server time: {si.CurrentTime()}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to connect to vCenter: {e}")
        raise VCenterConnectionError(
            host=config.vcenter.host,
            message=str(e),
            details={"error_type": type(e).__name__}
        )


def timed_data_collection() -> float:
    """
    Collect data from vCenter with timing information.
    
    Returns:
        float: Duration of data collection in seconds
        
    Raises:
        Exception: If data collection fails
    """
    logger = get_logger(__name__)
    
    start = time.time()
    logger.info("Starting data collection from vCenter")
    
    try:
        collect_and_store_metrics_main()
        end = time.time()
        duration = end - start
        
        # Save collection time for UI progress tracking
        with open("last_collection_time.txt", "w") as f:
            f.write(str(duration))
        
        logger.info(f"Data collection completed in {duration:.2f} seconds")
        return duration
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="vCenter DRS Compliance Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Collect data from vCenter
  python main.py check             # Test vCenter connectivity
  python main.py --log-level DEBUG # Run with debug logging
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='collect',
        choices=['collect', 'check'],
        help='Command to execute (default: collect)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = get_logger(__name__)
    
    logger.info("Starting vCenter DRS application")
    
    try:
        if args.command == 'check':
            # Test connectivity
            success = check_connectivity()
            if success:
                logger.info("Connectivity test passed")
                return 0
            else:
                logger.error("Connectivity test failed")
                return 1
        else:
            # Collect data
            start = time.time()
            duration = timed_data_collection()
            end = time.time()
            
            total_time = end - start
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            return 0
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())