import React from 'react';
import { Button, Tooltip, Space } from 'antd';
import { WarningOutlined, RobotOutlined } from '@ant-design/icons';

const ControlPanel = ({ 
  onMarkDefect,
  onAutoDetect,
  isDefect,
  loading
}) => {
  return (
    <Space>
      <Tooltip title="使用AI模型检测">
        <Button 
          icon={<RobotOutlined />} 
          onClick={onAutoDetect}
          loading={loading}
        >
          自动检测
        </Button>
      </Tooltip>
      <Tooltip title={isDefect ? '取消标记缺陷' : '标记为缺陷'}>
        <Button 
          type={isDefect ? 'primary' : 'default'} 
          danger={isDefect}
          onClick={onMarkDefect}
          icon={<WarningOutlined />}
          disabled={loading}
          style={isDefect ? {
            boxShadow: '0 2px 0 rgba(255,0,0,0.1)'
          } : undefined}
        >
          {isDefect ? '已标记缺陷' : '标记为缺陷'}
        </Button>
      </Tooltip>
    </Space>
  );
};

export default ControlPanel;
