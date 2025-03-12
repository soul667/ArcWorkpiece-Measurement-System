import React from 'react';
import { Upload, Button, message, Space, Switch } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import axios from 'axios';

const ModelPanel = ({ autoDetectEnabled, onToggleAutoDetect }) => {
  const customRequest = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    formData.append('model', file);
    
    try {
      await axios.post('/api/model/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      message.success('模型上传成功');
      onSuccess();
    } catch (error) {
      message.error('模型上传失败');
      onError(error);
    }
  };

  return (
    <Space>
      <Switch
        checked={autoDetectEnabled}
        onChange={onToggleAutoDetect}
        checkedChildren="自动检测开启"
        unCheckedChildren="自动检测关闭"
      />
      <Upload
        accept=".onnx"
        customRequest={customRequest}
        showUploadList={false}
      >
        <Button icon={<UploadOutlined />}>上传新模型</Button>
      </Upload>
    </Space>
  );
};

export default ModelPanel;
