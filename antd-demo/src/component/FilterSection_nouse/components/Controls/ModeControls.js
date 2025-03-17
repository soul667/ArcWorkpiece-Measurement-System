import React from 'react';
import PropTypes from 'prop-types';
import { Row, Col, Select, InputNumber, Button, Checkbox } from 'antd';
import { CloudOutlined, EyeOutlined } from '@ant-design/icons';

const ModeControls = ({
  xMode,
  yMode,
  zMode,
  nbNeighbors,
  stdRatio,
  showPointCloud,
  onModeChange,
  onNeighborsChange,
  onStdRatioChange,
  onShowPointCloudChange,
  onDenoise
}) => {
  return (
    <Row gutter={[16, 16]}>
      <Col>
        {/* 模式选择 */}
        <Select
          value={xMode}
          onChange={(value) => onModeChange('x', value)}
          style={{ width: 120, marginRight: 8 }}
          options={[
            { value: 'keep', label: 'X: 保留' },
            { value: 'remove', label: 'X: 移除' },
          ]}
        />
        <Select
          value={yMode}
          onChange={(value) => onModeChange('y', value)}
          style={{ width: 120, marginRight: 8 }}
          options={[
            { value: 'keep', label: 'Y: 保留' },
            { value: 'remove', label: 'Y: 移除' },
          ]}
        />
        <Select
          value={zMode}
          onChange={(value) => onModeChange('z', value)}
          style={{ width: 120, marginRight: 8 }}
          options={[
            { value: 'keep', label: 'Z: 保留' },
            { value: 'remove', label: 'Z: 移除' },
          ]}
        />

        {/* 去噪参数 */}
        <InputNumber
          value={nbNeighbors}
          onChange={onNeighborsChange}
          style={{ width: 120, marginRight: 4 }}
          min={1}
          max={1000}
          addonBefore="邻居点数"
        />
        <InputNumber
          value={stdRatio}
          onChange={onStdRatioChange}
          style={{ width: 120, marginRight: 8 }}
          min={0.1}
          max={2}
          step={0.1}
          addonBefore="标准差比例"
        />

        {/* 去噪按钮 */}
        <Button
          type="primary"
          danger
          icon={<CloudOutlined />}
          onClick={onDenoise}
          style={{ marginRight: 8 }}
        >
          去噪
        </Button>

        {/* 显示点云选项 */}
        <Checkbox
          checked={showPointCloud}
          onChange={(e) => onShowPointCloudChange(e.target.checked)}
          style={{ marginLeft: 8 }}
        >
          <EyeOutlined /> 显示点云
        </Checkbox>
      </Col>
    </Row>
  );
};

ModeControls.propTypes = {
  xMode: PropTypes.oneOf(['keep', 'remove']).isRequired,
  yMode: PropTypes.oneOf(['keep', 'remove']).isRequired,
  zMode: PropTypes.oneOf(['keep', 'remove']).isRequired,
  nbNeighbors: PropTypes.number.isRequired,
  stdRatio: PropTypes.number.isRequired,
  showPointCloud: PropTypes.bool.isRequired,
  onModeChange: PropTypes.func.isRequired,
  onNeighborsChange: PropTypes.func.isRequired,
  onStdRatioChange: PropTypes.func.isRequired,
  onShowPointCloudChange: PropTypes.func.isRequired,
  onDenoise: PropTypes.func.isRequired
};

export default ModeControls;
