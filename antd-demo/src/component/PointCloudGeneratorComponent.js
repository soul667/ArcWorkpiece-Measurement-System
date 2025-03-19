import React from 'react';
import { Card, Form, Input, InputNumber, Row, Col, Button, message, Space } from 'antd';
import axios from '../utils/axios';

const PointCloudGeneratorComponent = () => {
  const [form] = Form.useForm();

  return (
    <Card title="点云生成" bordered={false}>
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          noise_std: 0.01,
          arc_angle: 360,
          axis_direction: [0, 0, 1],
          axis_density: 500,
          arc_density: 100
        }}
      >
        <div style={{ 
          //background: '#fafafa', 
          padding: '12px 16px', 
          borderRadius: '4px',
          marginBottom: '16px'
        }}>
          <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>点云生成参数</div>
          
          {/* 噪声大小 */}
          <Row gutter={24}>
            <Col span={12}>
              <Form.Item
                label="噪声大小"
                name="noise_std"
                tooltip="添加到点云的随机噪声标准差"
              >
                <InputNumber
                  min={0}
                  max={0.1}
                  step={0.001}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            
            {/* 圆心角 */}
            <Col span={12}>
              <Form.Item
                label="圆心角(度)"
                name="arc_angle"
                tooltip="圆柱体的圆心角，360度为完整圆柱"
              >
                <InputNumber
                  min={0}
                  max={360}
                  style={{ width: '100%' }}
                  addonAfter="°"
                />
              </Form.Item>
            </Col>
          </Row>

          {/* 轴线方向 */}
          <Row gutter={24}>
            <Col span={24}>
              <Form.Item
                label="轴线方向"
                tooltip="圆柱体的轴线方向向量"
              >
                <Input.Group compact>
                  <Form.Item
                    name={['axis_direction', 0]}
                    noStyle
                  >
                    <InputNumber
                      placeholder="X"
                      style={{ width: '33%' }}
                    />
                  </Form.Item>
                  <Form.Item
                    name={['axis_direction', 1]}
                    noStyle
                  >
                    <InputNumber
                      placeholder="Y"
                      style={{ width: '33%' }}
                    />
                  </Form.Item>
                  <Form.Item
                    name={['axis_direction', 2]}
                    noStyle
                  >
                    <InputNumber
                      placeholder="Z"
                      style={{ width: '34%' }}
                    />
                  </Form.Item>
                </Input.Group>
              </Form.Item>
            </Col>
          </Row>

          {/* 密度参数 */}
          <Row gutter={24}>
            <Col span={12}>
              <Form.Item
                label="沿轴线密度"
                name="axis_density"
                tooltip="轴向上的点数"
              >
                <InputNumber
                  min={1}
                  max={1000}
                  style={{ width: '100%' }}
                  addonAfter="点"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="圆弧密度"
                name="arc_density"
                tooltip="圆周方向上的点数"
              >
                <InputNumber
                  min={1}
                  max={1000}
                  style={{ width: '100%' }}
                  addonAfter="点"
                />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ marginTop: '16px' }}>
            <Space>
              <Button 
                type="primary"
                onClick={() => {
                  form.validateFields().then(values => {
                    axios.post('/api/generate-point-cloud', values, {
                      responseType: 'blob'
                    }).then(response => {
                      const contentDisposition = response.headers['content-disposition'];
                      let filename = 'generated_cloud.ply';
                      if (contentDisposition) {
                        const filenameMatch = contentDisposition.match(/filename=(.+)/);
                        if (filenameMatch && filenameMatch.length > 1) {
                          filename = filenameMatch[1];
                        }
                      }

                      // 创建 Blob URL
                      const blob = new Blob([response.data], { type: 'application/octet-stream' });
                      const url = window.URL.createObjectURL(blob);
                      
                      // 创建下载链接
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = filename;
                      document.body.appendChild(link);
                      link.click();
                      
                      // 清理
                      document.body.removeChild(link);
                      window.URL.revokeObjectURL(url);
                      
                      message.success('点云生成并下载成功');
                    }).catch(error => {
                      // 如果是blob类型的错误响应，需要先读取内容
                      if (error.response?.data instanceof Blob) {
                        const reader = new FileReader();
                        reader.onload = () => {
                          try {
                            const errorData = JSON.parse(reader.result);
                            message.error('生成失败: ' + (errorData.detail || '未知错误'));
                          } catch {
                            message.error('生成失败: 未知错误');
                          }
                        };
                        reader.readAsText(error.response.data);
                      } else {
                        message.error('生成失败: ' + (error.message || '未知错误'));
                      }
                    });
                  });
                }}
              >
                生成并下载点云
              </Button>
            </Space>
          </div>
        </div>
      </Form>
    </Card>
  );
};

export default PointCloudGeneratorComponent;
