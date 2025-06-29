"""
Database Layer for vCenter DRS

This module provides database connectivity and schema management for the vCenter DRS system.
It handles connections to MySQL and provides methods for initializing the database schema.

The database stores:
- Clusters: vCenter clusters
- Hosts: ESXi hosts with cluster associations
- VMs: Virtual machines with host and dataset associations
- Datasets: Storage datastores
- Metrics: Performance metrics for VMs and hosts
"""

import os
import json
from typing import Optional, Dict, Any, Tuple, List
import mysql.connector
from mysql.connector import Error, MySQLConnection


class MetricsDB:
    """
    Database manager for vCenter DRS metrics and compliance data.
    
    This class handles all database operations including connection management,
    schema initialization, and data persistence for the vCenter DRS system.
    
    Attributes:
        host (str): MySQL server hostname
        user (str): MySQL username
        password (str): MySQL password
        database (str): MySQL database name
        conn (Optional[MySQLConnection]): Active database connection
    """
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        user: Optional[str] = None, 
        password: Optional[str] = None, 
        database: Optional[str] = None, 
        credentials_path: Optional[str] = None
    ) -> None:
        """
        Initialize the database connection parameters.
        
        Args:
            host: MySQL server hostname
            user: MySQL username
            password: MySQL password
            database: MySQL database name
            credentials_path: Path to JSON file containing database credentials
            
        Note:
            If credentials are not provided directly, they will be loaded from
            the credentials file. The credentials file should contain:
            - db_host: MySQL server address
            - db_user: MySQL username
            - db_password: MySQL password
            - db_database: MySQL database name
        """
        if not (host and user and password and database):
            if credentials_path is None:
                credentials_path = os.path.join(
                    os.path.dirname(__file__), '..', 'credentials.json'
                )
            
            with open(os.path.abspath(credentials_path), 'r') as f:
                creds: Dict[str, str] = json.load(f)
            
            self.host = creds['db_host']
            self.user = creds['db_user']
            self.password = creds['db_password']
            self.database = creds['db_database']
        else:
            self.host = host
            self.user = user
            self.password = password
            self.database = database
        
        self.conn: Optional[MySQLConnection] = None

    def connect(self) -> None:
        """
        Establish connection to the MySQL database.
        
        Raises:
            Error: If connection to MySQL fails
            
        Note:
            This method creates a new connection to the MySQL database.
            The connection is stored in self.conn for use by other methods.
        """
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print("Connected to MySQL database.")
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            self.conn = None

    def init_schema(self) -> None:
        """
        Initialize the database schema.
        
        This method creates all necessary tables if they don't exist:
        - clusters: vCenter clusters
        - hosts: ESXi hosts with cluster associations
        - datasets: Storage datastores
        - vms: Virtual machines with host and dataset associations
        - metrics: Performance metrics for VMs and hosts
        
        Raises:
            Exception: If schema initialization fails
        """
        if not self.conn:
            print("Not connected to database.")
            return
        
        cursor = self.conn.cursor()
        
        try:
            # Create clusters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE
                ) ENGINE=InnoDB;
            ''')
            
            # Create hosts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hosts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    cluster_id INT,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
                ) ENGINE=InnoDB;
            ''')
            
            # Create datasets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE,
                    description TEXT
                ) ENGINE=InnoDB;
            ''')
            
            # Create vms table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vms (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    host_id INT,
                    dataset_id INT,
                    FOREIGN KEY (host_id) REFERENCES hosts(id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                ) ENGINE=InnoDB;
            ''')
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    object_type ENUM('vm', 'host'),
                    object_id INT,
                    metric_name VARCHAR(255),
                    value FLOAT,
                    timestamp DATETIME,
                    INDEX idx_object (object_type, object_id),
                    INDEX idx_metric (metric_name, timestamp)
                ) ENGINE=InnoDB;
            ''')
            
            self.conn.commit()
            print("Database schema initialized.")
            
        except Exception as e:
            print(f"Error initializing schema: {e}")
            raise
        finally:
            cursor.close()

    def close(self) -> None:
        """
        Close the database connection.
        
        This method safely closes the MySQL connection if it exists.
        """
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def __enter__(self) -> 'MetricsDB':
        """Context manager entry point."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self.close()

# Example usage:
# db = MetricsDB()  # Loads from credentials.json by default
# db.connect()
# db.init_schema()
# db.close() 