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
        - exceptions: Exceptions for compliance rules
        
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
            
            # Create exceptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exceptions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    rule_type VARCHAR(64),
                    alias VARCHAR(255),
                    affected_vms TEXT,
                    cluster VARCHAR(255),
                    rule_hash VARCHAR(64),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
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

    def add_exception(self, exception_dict: dict) -> None:
        """
        Add an exception to the exceptions table.
        exception_dict should contain: rule_type, alias, affected_vms (list), cluster, rule_hash, reason (optional)
        """
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        import json as _json
        cursor.execute('''
            INSERT INTO exceptions (rule_type, alias, affected_vms, cluster, rule_hash, reason)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            exception_dict.get('rule_type'),
            exception_dict.get('alias'),
            _json.dumps(exception_dict.get('affected_vms', [])),
            exception_dict.get('cluster'),
            exception_dict.get('rule_hash'),
            exception_dict.get('reason', None)
        ))
        self.conn.commit()
        cursor.close()

    def get_exceptions(self) -> list:
        """
        Fetch all exceptions from the exceptions table.
        Returns a list of dicts.
        """
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM exceptions')
        rows = cursor.fetchall()
        cursor.close()
        import json as _json
        for row in rows:
            if 'affected_vms' in row and row['affected_vms']:
                try:
                    row['affected_vms'] = _json.loads(row['affected_vms'])
                except Exception:
                    row['affected_vms'] = []
        return rows

    def is_exception(self, violation_dict: dict) -> bool:
        """
        Check if a violation matches any exception in the table.
        Matching is based on rule_type, alias, affected_vms, and cluster.
        """
        exceptions = self.get_exceptions()
        import hashlib
        import json as _json
        def make_hash(d):
            # Use a stable hash of the relevant fields, normalized
            alias = (d.get('alias') or '').strip().lower()
            cluster = (d.get('cluster') or '').strip().lower()
            affected_vms = [vm.strip().lower() for vm in d.get('affected_vms', [])]
            relevant = {
                'rule_type': d.get('type'),
                'alias': alias,
                'affected_vms': sorted(affected_vms),
                'cluster': cluster
            }
            return hashlib.sha256(_json.dumps(relevant, sort_keys=True).encode()).hexdigest()
        v_hash = make_hash(violation_dict)
        for exc in exceptions:
            if exc.get('rule_hash') == v_hash:
                return True
        return False

# Example usage:
# db = MetricsDB()  # Loads from credentials.json by default
# db.connect()
# db.init_schema()
# db.close() 