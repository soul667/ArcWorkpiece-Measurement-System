import pytest
import mysql.connector
from mysql.connector import Error

class TestMySQLConnection:
    @pytest.fixture
    def db_config(self):
        return {
            'host': 'mysql',
            'port': 3306,
            'user': 'dev',
            'password': 'devpass',
            'database': 'arcworkpiece'
        }

    @pytest.fixture
    def connection(self, db_config):
        try:
            conn = mysql.connector.connect(**db_config)
            yield conn
            conn.close()
        except Error as e:
            pytest.fail(f"Failed to connect to MySQL: {e}")

    def test_connection(self, connection):
        """Test if we can connect to MySQL server"""
        assert connection.is_connected()
    
    def test_database_exists(self, connection):
        """Test if the specified database exists"""
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        database = cursor.fetchone()[0]
        cursor.close()
        assert database == "arcworkpiece"

    def test_crud_operations(self, connection):
        """Test Create, Read, Update, and Delete operations"""
        cursor = connection.cursor()
        
        try:
            # Create test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    value FLOAT
                )
            """)
            
            # Insert test data
            cursor.execute("""
                INSERT INTO test_table (name, value)
                VALUES (%s, %s)
            """, ("test_item", 123.45))
            connection.commit()
            
            # Read data
            cursor.execute("SELECT * FROM test_table WHERE name = %s", ("test_item",))
            result = cursor.fetchone()
            assert result is not None
            assert result[1] == "test_item"
            assert abs(result[2] - 123.45) < 0.001  # Compare floats with tolerance
            
            # Update data
            cursor.execute("""
                UPDATE test_table
                SET value = %s
                WHERE name = %s
            """, (456.78, "test_item"))
            connection.commit()
            
            # Verify update
            cursor.execute("SELECT value FROM test_table WHERE name = %s", ("test_item",))
            result = cursor.fetchone()
            assert abs(result[0] - 456.78) < 0.001
            
            # Delete data
            cursor.execute("DELETE FROM test_table WHERE name = %s", ("test_item",))
            connection.commit()
            
            # Verify deletion
            cursor.execute("SELECT * FROM test_table WHERE name = %s", ("test_item",))
            result = cursor.fetchone()
            assert result is None

        finally:
            # Clean up: drop test table
            cursor.execute("DROP TABLE IF EXISTS test_table")
            connection.commit()
            cursor.close()

    def test_error_handling(self, connection):
        """Test error handling for invalid queries"""
        cursor = connection.cursor()
        
        # Test invalid table name
        with pytest.raises(Error):
            cursor.execute("SELECT * FROM nonexistent_table")
        
        # Test invalid column name
        cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INT)")
        with pytest.raises(Error):
            cursor.execute("SELECT nonexistent_column FROM test_table")
        
        # Clean up
        cursor.execute("DROP TABLE IF EXISTS test_table")
        connection.commit()
        cursor.close()

if __name__ == "__main__":
    pytest.main([__file__])
