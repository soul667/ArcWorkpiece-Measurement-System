import React, { useState, useEffect } from 'react';
import { Card, Typography, Button, Upload, message, Progress, Spin, Divider, List, Image, Row, Col, Popconfirm } from 'antd';
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
  const [tempClouds, setTempClouds] = useState([]);
  const [loading, setLoading] = useState(false);

  const [isDragging, setIsDragging] = useState(false);

  // 获取暂存点云列表
  const fetchTempClouds = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/temp-save/list');
      if (response.data.status === 'success') {
        setTempClouds(response.data.data);
      }
    } catch (error) {
      message.error('获取暂存点云列表失败');
    } finally {
      setLoading(false);
    }
  };

  // 暂存当前点云
  const handleStore = async () => {
    try {
      const response = await axios.post('/api/temp-save/store');
      if (response.data.status === 'success') {
        message.success('点云暂存成功');
        fetchTempClouds();
      }
    } catch (error) {
      message.error('点云暂存失败');
    }
  };

  // 加载点云
  const handleLoad = async (timestamp) => {
    try {
      const response = await axios.get(`/api/temp-save/${timestamp}/load`);
      if (response.data.status === 'success') {
        message.success('点云加载成功');
      }
    } catch (error) {
      message.error('点云加载失败');
    }
  };

  // 删除点云
  const handleDelete = async (timestamp) => {
    try {
      const response = await axios.delete(`/api/temp-save/${timestamp}`);
      if (response.data.status === 'success') {
        message.success('点云删除成功');
        fetchTempClouds();
      }
    } catch (error) {
      message.error('点云删除失败');
    }
  };

  // 组件加载时获取暂存列表
  useEffect(() => {
    fetchTempClouds();
  }, []);

  const draggerStyle = {
    //background: '#fafafa',
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

              const response = await axios.post('/api/point-cloud/upload', formData, {
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

      {/* 暂存功能 */}
      <Divider>点云暂存</Divider>
      
      <div style={{ marginBottom: '16px' }}>
        <Button 
          type="primary" 
          onClick={handleStore}
          style={{ marginRight: '8px' }}
        >
          暂存当前点云
        </Button>
      </div>

      <List
        loading={loading}
        dataSource={tempClouds}
        renderItem={item => (
          <List.Item
            key={item.id}
            style={{
              //background: '#fafafa',
              marginBottom: '8px',
              padding: '16px',
              borderRadius: '4px'
            }}
            actions={[
              <Button type="primary" onClick={() => handleLoad(item.timestamp)}>
                加载点云
              </Button>,
              <Popconfirm
                title="确定要删除这个暂存点云吗？"
                onConfirm={() => handleDelete(item.timestamp)}
                okText="确定"
                cancelText="取消"
              >
                <Button type="text" danger icon={<DeleteOutlined />}>
                  删除
                </Button>
              </Popconfirm>
            ]}
          >
            <div style={{ width: '100%' }}>
              <div style={{ 
                marginBottom: '8px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <Typography.Text strong>
                  暂存时间: {new Date(item.createdAt).toLocaleString()}
                </Typography.Text>
              </div>
              
              <Row gutter={16}>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Typography.Text type="secondary">XY视图</Typography.Text>
                    <Image
                      src={`/api/files/img/${item.views.xy}?t=${Date.now()}`}
                      alt="XY视图"
                      style={{ width: '100%', maxHeight: '150px', objectFit: 'contain' }}
                    />
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Typography.Text type="secondary">YZ视图</Typography.Text>
                    <Image
                      src={`/api/files/img/${item.views.yz}?t=${Date.now()}`}
                      alt="YZ视图"
                      style={{ width: '100%', maxHeight: '150px', objectFit: 'contain' }}
                    />
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Typography.Text type="secondary">XZ视图</Typography.Text>
                    <Image
                      src={`/api/files/img/${item.views.xz}?t=${Date.now()}`}
                      alt="XZ视图"
                      style={{ width: '100%', maxHeight: '150px', objectFit: 'contain' }}
                    />
                  </div>
                </Col>
              </Row>
            </div>
          </List.Item>
        )}
      />

      <style jsx="true">{`
        .upload-card .ant-upload-drag {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
        }
        .upload-card .ant-upload-drag:hover {
          border-color: #1890ff;
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
          border-radius: 4px;
        }
        .upload-card .ant-upload-list-item {
          transition: all 0.3s;
        }
      `}</style>
    </Card>
  );
};

export default UploadSection;
