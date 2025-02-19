import React from 'react';
import { Form, InputNumber, Select, Card, Row, Col, Button, message, Divider } from 'antd';
const { Option } = Select;

const ParamsSettingComponent = () => {
  const [form] = Form.useForm();

  const handlePreprocess = () => {
    form.validateFields().then((values) => {
      message.success('参数提交成功！');
      console.log('提交数据:', values);
    });
  };

  const formItemLayout = {
    labelCol: { style: { fontWeight: 500 } }
  };

  return (
    <Card title="参数设置" bordered={false}>
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          cylinderMethod: 'RANSAC',
          circleFitMethod: 'Hyper',
          normalNeighbors: 20,
          ransacThreshold: 0.1,
          maxRadius: 11,
          minRadius: 6,
        }}
      >
        <div style={{ 
          background: '#fafafa', 
          padding: '12px 16px', 
          borderRadius: '4px',
          marginBottom: '16px'
        }}>
          <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>算法方法设置</div>
          <Row gutter={24}>
            <Col span={12}>
              <Form.Item 
                {...formItemLayout}
                label="寻找圆柱轴方法" 
                name="cylinderMethod"
                tooltip="用于确定圆柱轴线的算法方法"
              >
                <Select>
                  <Option value="RANSAC">RANSAC</Option>
                  <Option value="NormalPCA">NormalPCA</Option>
                  <Option value="RobPCA">RobPCA</Option>
                  <Option value="LeastSquares">最小二乘</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item 
                {...formItemLayout}
                label="圆拟合方法" 
                name="circleFitMethod"
                tooltip="用于拟合圆形的算法方法"
              >
                <Select>
                  <Option value="Hyper">Hyper</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </div>

        <div style={{ 
          background: '#fafafa', 
          padding: '12px 16px', 
          borderRadius: '4px',
          marginBottom: '16px'
        }}>
          <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>参数阈值设置</div>
          <Row gutter={24}>
            <Col span={8}>
              <Form.Item 
                {...formItemLayout}
                label="求法线邻近点数量" 
                name="normalNeighbors"
                tooltip="计算点云法线时使用的邻近点数量"
              >
                <InputNumber 
                  min={1} 
                  max={100} 
                  style={{ width: '100%' }}
                  addonAfter="个" 
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item 
                {...formItemLayout}
                label="RANSAC阈值" 
                name="ransacThreshold"
                tooltip="RANSAC算法的距离阈值"
              >
                <InputNumber 
                  min={0.01} 
                  max={1.0} 
                  step={0.01} 
                  style={{ width: '100%' }} 
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={24}>
            <Col span={8}>
              <Form.Item 
                {...formItemLayout}
                label="最小半径" 
                name="minRadius"
                tooltip="圆柱的最小半径限制"
              >
                <InputNumber 
                  min={1} 
                  max={50} 
                  style={{ width: '100%' }}
                  addonAfter="mm" 
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item 
                {...formItemLayout}
                label="最大半径" 
                name="maxRadius"
                tooltip="圆柱的最大半径限制"
              >
                <InputNumber 
                  min={1} 
                  max={50} 
                  style={{ width: '100%' }}
                  addonAfter="mm" 
                />
              </Form.Item>
            </Col>
          </Row>
        </div>

        <div>
          <Button type="primary" onClick={handlePreprocess}>
            确认设置
          </Button>
        </div>
      </Form>
    </Card>
  );
};

export default ParamsSettingComponent;
