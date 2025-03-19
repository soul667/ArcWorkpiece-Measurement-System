DROP TABLE IF EXISTS measurement_history;
CREATE TABLE measurement_history (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,                    -- 用户ID
    cloud_name VARCHAR(255) NOT NULL,        -- 点云文件名称
    timestamp VARCHAR(50) NOT NULL,          -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    
    -- 测量结果
    radius FLOAT NOT NULL,                   -- 拟合半径
    axis_vector_x FLOAT NOT NULL,           -- 轴线方向向量x分量
    axis_vector_y FLOAT NOT NULL,           -- 轴线方向向量y分量
    axis_vector_z FLOAT NOT NULL,           -- 轴线方向向量z分量
    axis_point_x FLOAT NOT NULL,            -- 轴线上一点x坐标
    axis_point_y FLOAT NOT NULL,            -- 轴线上一点y坐标
    axis_point_z FLOAT NOT NULL,            -- 轴线上一点z坐标
    
    -- 投影图
    original_projection VARCHAR(255),        -- 原始投影图路径
    axis_projection VARCHAR(255),            -- 轴向投影图路径
    
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 添加索引
CREATE INDEX history_user_id_idx ON measurement_history(user_id);
CREATE INDEX history_timestamp_idx ON measurement_history(timestamp);
CREATE INDEX history_created_at_idx ON measurement_history(created_at);
