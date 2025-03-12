import React from 'react';
import { Select, Space } from 'antd';

const AxisSelector = ({ selectedAxis, onAxisChange }) => {
  const axisOptions = [
    { label: 'X轴', value: 'x' },
    { label: 'Y轴', value: 'y' },
    { label: 'Z轴', value: 'z' },
  ];

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Select
        value={selectedAxis}
        onChange={onAxisChange}
        options={axisOptions}
        style={{ width: '100%' }}
        placeholder="选择分组轴"
      />
    </Space>
  );
};

export default AxisSelector;
