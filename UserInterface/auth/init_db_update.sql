-- 添加线条设置表
CREATE TABLE IF NOT EXISTS line_settings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    point_size INT DEFAULT 3,
    defect_lines TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 添加索引
CREATE INDEX line_settings_user_id_idx ON line_settings(user_id);
CREATE INDEX line_settings_created_at_idx ON line_settings(created_at);
