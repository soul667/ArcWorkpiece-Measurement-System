import React from 'react';
import PropTypes from 'prop-types';
import { List, Button, Typography } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';

const { Text } = Typography;

const RegionList = ({
  regions,
  type,
  onDelete,
  header
}) => {
  return (
    <List
      size="small"
      header={<Text strong>{header || `${type === 'x' ? '垂直' : '水平'}区域 (${type.toUpperCase()}):`}</Text>}
      dataSource={regions.map((region, index) => ({
        region,
        index
      }))}
      renderItem={({ region, index }) => (
        <List.Item
          actions={[
            <Button 
              type="link" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => onDelete(index)}
            >
              删除
            </Button>
          ]}
        >
          区域 {index + 1}: [{region[0].toFixed(2)}, {region[1].toFixed(2)}]
        </List.Item>
      )}
      style={{ 
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        borderRadius: '4px',
        padding: '8px'
      }}
    />
  );
};

RegionList.propTypes = {
  regions: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)).isRequired,
  type: PropTypes.oneOf(['x', 'y']).isRequired,
  onDelete: PropTypes.func.isRequired,
  header: PropTypes.string
};

export default RegionList;
