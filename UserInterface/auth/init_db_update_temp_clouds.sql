-- 添加临时点云存储表
CREATE TABLE IF NOT EXISTS temp_clouds (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    timestamp VARCHAR(50) NOT NULL,  -- 时间戳目录名
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 添加索引
DROP INDEX IF EXISTS temp_clouds_user_id_idx ON temp_clouds;
DROP INDEX IF EXISTS temp_clouds_created_at_idx ON temp_clouds;
DROP INDEX IF EXISTS temp_clouds_timestamp_idx ON temp_clouds;

CREATE INDEX temp_clouds_user_id_idx ON temp_clouds(user_id);
CREATE INDEX temp_clouds_created_at_idx ON temp_clouds(created_at);
CREATE INDEX temp_clouds_timestamp_idx ON temp_clouds(timestamp);
