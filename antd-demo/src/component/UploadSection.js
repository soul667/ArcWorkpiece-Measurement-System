import React, { useState } from 'react';
import { Card, Typography, Button, Upload, message, Progress, Spin } from 'antd';
import { 
  InboxOutlined, 
  CloudUploadOutlined, 
  DeleteOutlined, 
  FileOutlined,
  CheckCircleFilled,
  CloseCircleFilled 
} from '@ant-design/icons';
import axios from '../utils/axios';

const { Text } = Typography;

const UploadSection = () => {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [isDragging, setIsDragging] = useState(false);

  const draggerStyle = {
    background: '#fafafa',
    border: `1px dashed ${isDragging ? '#1890ff' : '#d9d9d9'}`,
    borderRadius: '8px',
    padding: '24px',
    transition: 'all 0.3s',
    ...(isDragging && {
      background: '#f0f7ff',
      borderColor: '#1890ff',
      transform: 'scale(1.01)'
    })
  };

  return (
    <Card 
      title="点云上传" 
      bordered={false}
      className="upload-card"
    >
      <Upload.Dragger
        accept=".ply"
        multiple={false}
        showUploadList={{
          showPreviewIcon: false,
          showDownloadIcon: false,
          showRemoveIcon: true,
          removeIcon: <DeleteOutlined style={{ color: '#ff4d4f' }} />
        }}
        maxCount={1}
        itemRender={(originNode, file) => (
          <div style={{
            padding: '8px 0',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <FileOutlined style={{ marginRight: 8, color: '#1890ff' }} />
              <span>{file.name}</span>
            </div>
            {file.status === 'done' && (
              <CheckCircleFilled style={{ color: '#52c41a' }} />
            )}
            {file.status === 'error' && (
              <CloseCircleFilled style={{ color: '#ff4d4f' }} />
            )}
          </div>
        )}
        progress={{
          strokeColor: {
            '0%': '#108ee9',
            '100%': '#87d068',
          },
          strokeWidth: 3,
          format: percent => `${parseFloat(percent.toFixed(2))}%`
        }}
        beforeUpload={async (file) => {
            // 文件类型检查
            const isPLY = file.name.endsWith('.ply');
            if (!isPLY) {
              message.error('只能上传PLY格式的点云文件！');
              return false;
            }

            try {
              setUploading(true);
              setUploadStatus(null);

              // 获取最新参数设置
              const settingsResponse = await axios.get('/api/settings/latest');
              const cylinderSettings = settingsResponse.data.data.cylinderSettings || {};
              
              const formData = new FormData();
              formData.append('file', file);
              formData.append('actual_speed', cylinderSettings.actualSpeed || 100);
              formData.append('acquisition_speed', cylinderSettings.acquisitionSpeed || 100);

              const response = await axios.post('/upload', formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
                timeout: 600000,
                onUploadProgress: (progressEvent) => {
                  const percent = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                  );
                  setUploading(true);
                  setUploadStatus(`上传进度: ${percent}%`);
                }
              });

              setUploading(false);
              if (response.data.error) {
                setUploadStatus('error');
                message.error(response.data.error);
              } else {
                setUploadStatus('success');
                message.success(`${file.name} 上传成功`);
              }
            } catch (error) {
              setUploading(false);
              setUploadStatus('error');
              message.error(error.response?.data?.error || '文件上传失败，请重试');
            }

            return false; // 阻止默认上传行为
          }}
        style={draggerStyle}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        onDrop={() => setIsDragging(false)}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined style={{ 
            fontSize: '48px', 
            color: isDragging ? '#1890ff' : '#40a9ff',
            transform: isDragging ? 'scale(1.1)' : 'scale(1)',
            transition: 'all 0.3s'
          }} />
        </p>
        <p className="ant-upload-text" style={{ 
          fontSize: '16px', 
          marginTop: '16px',
          fontWeight: isDragging ? 500 : 400,
          color: isDragging ? '#1890ff' : 'inherit'
        }}>
          点击或拖拽文件到此区域上传点云文件
        </p>
        <p className="ant-upload-hint" style={{ color: '#666' }}>
          仅支持PLY格式文件
        </p>
        {uploading && (
          <div style={{ marginTop: '16px' }}>
            <Spin spinning={true} />
            <Text style={{ marginLeft: '8px', color: '#1890ff' }}>
              {uploadStatus}
            </Text>
            <Progress
              percent={parseInt(uploadStatus?.match(/\d+/) || 0)}
              size="small"
              status="active"
              style={{ marginTop: '8px' }}
            />
          </div>
        )}
        
        {uploadStatus === 'success' && !uploading && (
          <div style={{ marginTop: '16px' }}>
            <Text type="success">
              <CloudUploadOutlined style={{ marginRight: '8px' }} />
              文件上传成功！
            </Text>
          </div>
        )}
        
        {uploadStatus === 'error' && !uploading && (
          <Text type="danger" style={{ marginTop: '16px', display: 'block' }}>
            文件上传失败，请确保上传有效的PLY格式文件。
          </Text>
        )}
      </Upload.Dragger>
      <style jsx="true">{`
        .upload-card .ant-upload-drag {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
        }
        .upload-card .ant-upload-drag:hover {
          border-color: #1890ff;
          background: #f0f7ff;
        }
        .upload-card .ant-progress-bg {
          transition: all 0.4s cubic-bezier(0.08, 0.82, 0.17, 1);
        }
        .upload-card .ant-upload-drag-icon {
          margin-bottom: 16px;
        }
        .upload-card .ant-progress {
          width: 90%;
          margin: 0 auto;
        }
        .upload-card .ant-btn:hover {
          transform: translateY(-1px);
          box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .upload-card .ant-upload-list {
          margin-top: 16px;
          padding: 16px;
          background: #fafafa;
          border-radius: 4px;
        }
        .upload-card .ant-upload-list-item {
          transition: all 0.3s;
        }
        .upload-card .ant-upload-list-item:hover {
          background: #f0f7ff;
        }
      `}</style>
    </Card>
  );
};

export default UploadSection;
