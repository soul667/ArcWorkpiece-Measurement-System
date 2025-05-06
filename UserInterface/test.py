import time
import mysql.connector
from mysql.connector import Error

for _ in range(10):
    try:
        connection = mysql.connector.connect(
            host="mysql",
            port=3306,
            user="dev",
            password="devpass",
            database="arcworkpiece",
            pool_name="mypool",
            pool_size=5
        )
        print("✅ 成功连接数据库")
        break
    except Error as e:
        print(f"❌ 连接数据库失败，错误: {e}")
        time.sleep(2)
else:
    raise RuntimeError("⛔ 无法连接到数据库，请检查配置或容器状态")
