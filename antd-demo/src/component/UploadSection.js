import React, { useState } from 'react';
import { Card, Typography, Button, Upload, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import axios from '../utils/axios';

const { Text } = Typography;

const UploadSection = () => {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [hasUploadedCloud, setHasUploadedCloud] = useState(false);

  const handleShowCloud = () => {
    // 显示点云的逻辑
    message.success('显示点云');
  };

  return (
    <Card title="点云上传（仅支持PLY）" bordered={false}>
      <div style={{ display: 'flex', gap: '8px' }}>
        <Upload
          accept=".ply"
          multiple={false}
          showUploadList={false}
          beforeUpload={async (file) => {
            const isPLY = file.name.endsWith('.ply');
            if (!isPLY) {
              message.error('只能上传PLY格式的点云文件！');
              return false;
            }

            try {
              setUploading(true);
              setUploadStatus(null);

              const formData = new FormData();
              formData.append('file', file);

              const response = await axios.post('/upload', formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                }
              });

              setUploading(false);
              if (response.data.error) {
                setUploadStatus('error');
                message.error(response.data.error);
              } else {
                setUploadStatus('success');
                message.success(`${file.name} 上传成功`);
                setHasUploadedCloud(true);
              }
            } catch (error) {
              setUploading(false);
              setUploadStatus('error');
              message.error(error.response?.data?.error || '文件上传失败，请重试');
            }

            return false; // 阻止默认上传行为
          }}
        >
          <Button icon={<UploadOutlined />} loading={uploading}>
            {uploading ? '正在上传...' : '点击上传点云文件'}
          </Button>
        </Upload>
        {uploadStatus === 'success' && (
          <Text type="success" style={{ marginLeft: 8 }}>
            文件上传成功！
          </Text>
        )}
        {uploadStatus === 'error' && (
          <Text type="danger" style={{ marginLeft: 8 }}>
            文件上传失败，请确保上传有效的PLY格式文件。
          </Text>
        )}
        <Button onClick={handleShowCloud} disabled={!hasUploadedCloud}>
          显示点云
        </Button>
      </div>
    </Card>
  );
};

export default UploadSection;
