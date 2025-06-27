import os
import json
import mysql.connector
from mysql.connector import Error

class MetricsDB:
    def __init__(self, host=None, user=None, password=None, database=None, credentials_path=None):
        if not (host and user and password and database):
            if credentials_path is None:
                credentials_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
            with open(os.path.abspath(credentials_path), 'r') as f:
                creds = json.load(f)
            self.host = creds['db_host']
            self.user = creds['db_user']
            self.password = creds['db_password']
            self.database = creds['db_database']
        else:
            self.host = host
            self.user = user
            self.password = password
            self.database = database
        self.conn = None

    def connect(self):
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

    def init_schema(self):
        if not self.conn:
            print("Not connected to database.")
            return
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) UNIQUE
            ) ENGINE=InnoDB;
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hosts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                cluster_id INT,
                FOREIGN KEY (cluster_id) REFERENCES clusters(id)
            ) ENGINE=InnoDB;
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) UNIQUE,
                description TEXT
            ) ENGINE=InnoDB;
        ''')
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
        cursor.close()
        print("Database schema initialized.")

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

# Example usage:
# db = MetricsDB()  # Loads from credentials.json by default
# db.connect()
# db.init_schema()
# db.close() 