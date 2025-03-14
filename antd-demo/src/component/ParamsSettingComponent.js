import React, { useState, useEffect } from 'react';
import axios from '../utils/axios';
import { Form, InputNumber, Select, Card, Row, Col, Button, message, Divider, Tabs, Input, Space, List, Collapse, Popconfirm, Typography } from 'antd';
import { DeleteOutlined, CaretRightOutlined, UnorderedListOutlined } from '@ant-design/icons';
const { Option } = Select;
const { TabPane } = Tabs;

// 默认设置
const defaultArcSettings = {
  arcMethod: 'HyperFit',
  arcNormalNeighbors: 20,
  arcMaxRadius: 12,
  arcMinRadius: 6,
  learningRate: 0.01,
  gradientMaxIterations: 1000,
  tolerance: 1e-6,
  fitIterations: 50,      // 拟合迭代次数n
  samplePercentage: 50    // 采样百分比m
};

const defaultCylinderSettings = {
  cylinderMethod: 'NormalRANSAC',
  normalNeighbors: 20,
  ransacThreshold: 0.01,
  maxIterations: 1000,
  normalDistanceWeight: 0.1,
  maxRadius: 11,
  minRadius: 6,
  axisOrientation: 'x',
  actualSpeed: 100,
  acquisitionSpeed: 100
};

const ParamsSettingComponent = () => {
  const [form] = Form.useForm();
  const [arcForm] = Form.useForm();
  const [selectedMethod, setSelectedMethod] = useState('NormalRANSAC');
  const [selectedArcMethod, setSelectedArcMethod] = useState('HyperFit');
  const [savedSettings, setSavedSettings] = useState([]);
  // 更新名称
  const handleUpdateName = async (id, newName) => {
    try {
      const response = await axios.put(
        `/api/settings/${id}`,
        { name: newName }
      );
      if (response.data.status === 'success') {
        message.success('更新成功');
        fetchSavedSettings(); // 刷新列表
      }
    } catch (error) {
      message.error('更新失败');
    }
  };

  // 删除设置
  const handleDelete = async (item) => {
    try {
      const response = await axios.delete(`/api/settings/${item.id}`);
      if (response.data.status === 'success') {
        message.success('删除成功');
        fetchSavedSettings(); // 刷新列表
      }
    } catch (error) {
      message.error('删除失败');
    }
  };

  // 加载最新设置
  useEffect(() => {
    const fetchLatestSettings = async () => {
      try {
        const response = await axios.get('/api/settings/latest');
        console.log('最新设置:', response.data);
        if (response.data.status === 'success') {
          const { cylinderSettings, arcSettings } = response.data.data;
          // 如果有保存的设置就使用，没有就使用默认值
          form.setFieldsValue(cylinderSettings || defaultCylinderSettings);
          arcForm.setFieldsValue(arcSettings || defaultArcSettings);
        } else {
          // 如果没有保存的设置，使用默认值
          form.setFieldsValue(defaultCylinderSettings);
          arcForm.setFieldsValue(defaultArcSettings);
        }
      } catch (error) {
        console.error('获取最新设置失败，使用默认值');
        // 发生错误时使用默认值
        form.setFieldsValue(defaultCylinderSettings);
        arcForm.setFieldsValue(defaultArcSettings);
      }
    };

    fetchLatestSettings();
  }, [form, arcForm]);

  const handleMethodChange = (value) => {
    setSelectedMethod(value);
  };

  const handleArcMethodChange = (value) => {
    setSelectedArcMethod(value);
  };

  // 获取保存的设置列表
  const fetchSavedSettings = async () => {
    try {
      const response = await axios.get('/api/settings/list');
      if (response.data.status === 'success') {
        setSavedSettings(response.data.data);
      }
    } catch (error) {
      console.error('获取设置列表失败:', error);
    }
  };

  // 加载设置
  const loadSetting = async (id) => {
    try {
      const response = await axios.get(`/api/settings/${id}`);
      if (response.data.status === 'success') {
        const { cylinderSettings, arcSettings } = response.data.data;
        form.setFieldsValue(cylinderSettings);
        arcForm.setFieldsValue(arcSettings);
        message.success('设置加载成功');
      }
    } catch (error) {
      console.error('加载设置失败:', error);
    }
  };

  // 保存设置
  const handleSaveSetting = async () => {
    try {
      // 同时验证两个表单
      const [cylinderValues, arcValues] = await Promise.all([
        form.validateFields(),
        arcForm.validateFields()
      ]);

      // 确保使用至少包含默认值的数据
      const mergedArcValues = {
        ...defaultArcSettings,
        ...arcValues
      };

      // 生成时间戳作为设置名称
      const timestamp = new Date().toLocaleString('zh-CN').replace(/[\/\s:]/g, '');
      const data = {
        name: `设置_${timestamp}`,
        cylinderSettings: cylinderValues,
        arcSettings: mergedArcValues
      };

      // 打印调试信息
      console.log('准备保存的设置数据:', JSON.stringify(data, null, 2));

      const response = await axios.post('/api/settings/save', data);

      if (response.data.status === 'success') {
        message.success('设置保存成功');
        fetchSavedSettings();  // 刷新设置列表
      } else {
        message.error(response.data.error || '保存失败');
      }
    } catch (error) {
      console.error('保存设置失败:', error);
      if (error.errorFields) {
        // 表单验证错误
        if (error.errorFields.some(field => field.name[0].startsWith('arc'))) {
          message.error('请完成圆弧拟合参数的设置');
        } else {
          message.error('请完成圆柱轴线参数的设置');
        }
      } else if (error.response) {
        // 服务器返回的错误
        message.error(error.response.data.error || '保存失败，请检查数据是否完整');
      } else {
        // 其他错误
        message.error('保存失败，请重试');
      }
    }
  };

  // 在组件挂载时获取已保存的设置列表
  useEffect(() => {
    fetchSavedSettings();
  }, []);

  // 处理投影图像显示
  const handleProjectionImage = (data) => {
    const imgContainer = document.getElementById('projection-container');
    imgContainer.style.display = 'block';  // 始终显示容器
    
    // 移除旧内容
    imgContainer.querySelector('img')?.remove();
    imgContainer.querySelector('.loading-indicator')?.remove();
    
    if (data.projectionImage) {

      // 创建新图像并添加加载中状态
      const imgElement = document.createElement('img');
      imgElement.style.maxWidth = '100%';
      imgElement.style.maxHeight = '400px';  // 限制最大高度
      imgElement.style.objectFit = 'contain';
      imgElement.style.transition = 'opacity 0.3s';
      imgElement.style.opacity = '0';
      
      // 添加加载指示器
      const loadingDiv = document.createElement('div');
      loadingDiv.style.textAlign = 'center';
      loadingDiv.style.padding = '20px';
      loadingDiv.textContent = '图像加载中...';
      
      // 移除旧内容
      imgContainer.querySelector('img')?.remove();
      imgContainer.querySelector('.loading-indicator')?.remove();
      
      // 添加新元素
      imgContainer.appendChild(loadingDiv);
      
      // 设置图片加载事件
      imgElement.onload = () => {
        loadingDiv.remove();
        imgElement.style.opacity = '1';
      };
      
      // 设置图片源
      imgElement.src = data.projectionImage;
      imgContainer.appendChild(imgElement);
    }
  };

  const handlePreprocess = () => {
    form.validateFields().then(async (values) => {
      try {
        const processData = {
          cylinder_method: values.cylinderMethod,
          normal_neighbors: values.normalNeighbors,
          min_radius: values.minRadius,
          max_radius: values.maxRadius,
          ransac_threshold: values.ransacThreshold,
          max_iterations: values.maxIterations,
          normal_distance_weight: values.normalDistanceWeight,
          axis_orientation: values.axisOrientation,
          actual_speed: values.actualSpeed,
          acquisition_speed: values.acquisitionSpeed
        };

        const response = await axios.post('/process', processData);
        const data = response.data;
        
        if (data.status === 'success') {
          message.success('参数提交成功！');
          if (data.radius) {
            message.info(`拟合半径: ${data.radius.toFixed(2)}mm`);
          }
          handleProjectionImage(data);
        }
      } catch (error) {
        console.error('Error:', error);
      }
    });
  };

  const handleArcFit = async () => {
    arcForm.validateFields().then(async (values) => {
      try {
        const arcFitData = {
          arc_method: values.arcMethod,
          normal_neighbors: values.arcNormalNeighbors,
          min_radius: values.arcMinRadius,
          max_radius: values.arcMaxRadius,
        };

        if (values.arcMethod === 'GradientDescent') {
          arcFitData.gradient_params = {
            learning_rate: values.learningRate,
            max_iterations: values.gradientMaxIterations,
            tolerance: values.tolerance
          };
        }

        const response = await axios.post('/fit_arc', arcFitData);
        const data = response.data;
        
        if (data.status === 'success') {
          message.success('圆弧拟合成功！');
          if (data.radius) {
            message.info(`圆弧拟合半径: ${data.radius.toFixed(2)}mm`);
          }
          if (data.center) {
            message.info(`圆弧拟合中心: (${data.center[0].toFixed(2)}, ${data.center[1].toFixed(2)})`);
          }
        }
      } catch (error) {
        console.error('Error:', error);
      }
    });
  };

  const handleSequentialFit = async () => {
    try {
      // First do cylinder fitting
      await form.validateFields().then(async (values) => {
        const processData = {
          cylinder_method: values.cylinderMethod,
          normal_neighbors: values.normalNeighbors,
          min_radius: values.minRadius,
          max_radius: values.maxRadius,
          ransac_threshold: values.ransacThreshold,
          max_iterations: values.maxIterations,
          normal_distance_weight: values.normalDistanceWeight,
          axis_orientation: values.axisOrientation,
          actual_speed: values.actualSpeed,
          acquisition_speed: values.acquisitionSpeed
        };

        const response = await axios.post('/process', processData);
        const data = response.data;
        
        if (data.status !== 'success') {
          throw new Error(data.error || '轴线拟合失败');
        }
        message.success('轴线拟合完成');
        handleProjectionImage(data);
      });

      // Then do arc fitting
      await arcForm.validateFields().then(async (values) => {
        const arcFitData = {
          arc_method: values.arcMethod,
          normal_neighbors: values.arcNormalNeighbors,
          min_radius: values.arcMinRadius,
          max_radius: values.arcMaxRadius,
        };

        if (values.arcMethod === 'GradientDescent') {
          arcFitData.gradient_params = {
            learning_rate: values.learningRate,
            max_iterations: values.gradientMaxIterations,
            tolerance: values.tolerance
          };
        }

        const response = await axios.post('/fit_arc', arcFitData);
        const data = response.data;

        if (data.status !== 'success') {
          throw new Error(data.error || '圆弧拟合失败');
        }
        message.success('圆弧拟合完成');
      });

      message.success('序列拟合完成');
    } catch (error) {
      console.error('Error:', error);
      message.error('拟合失败：' + error.message);
    }
  };

  const formItemLayout = {
    labelCol: { style: { fontWeight: 500 } }
  };

  return (
    <Card title="参数设置" bordered={false} id="cylinder-params-card">
      {/* 投影图像显示区域 */}
      <div id="projection-container" style={{ 
        marginBottom: '24px',
        textAlign: 'center',
        background: '#fafafa',
        border: '1px solid #d9d9d9',
        borderRadius: '4px',
        padding: '16px',
        display: 'none',  // 初始隐藏
        minHeight: '200px',
        transition: 'all 0.3s'
      }}>
        <h3 style={{ color: '#1890ff', margin: '0 0 16px 0' }}>圆柱体点云投影视图</h3>
      </div>

      <Tabs defaultActiveKey="cylinder">
        <TabPane tab="圆柱轴线拟合" key="cylinder">
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              cylinderMethod: 'NormalRANSAC',
              normalNeighbors: 20,
              ransacThreshold: 0.01,
              maxIterations: 1000,
              normalDistanceWeight: 0.1,
              maxRadius: 11,
              minRadius: 6,
              axisOrientation: 'x',
              actualSpeed: 100,
              acquisitionSpeed: 100,
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
                <Col span={24}>
                  <Form.Item 
                    {...formItemLayout}
                    label="寻找圆柱轴方法" 
                    name="cylinderMethod"
                    tooltip="用于确定圆柱轴线的算法方法"
                  >
                    <Select onChange={handleMethodChange}>
                      <Option value="NormalRANSAC">基于法向量的RANSAC</Option>
                      <Option value="NormalLeastSquares">基于法向量的最小二乘法</Option>
                      <Option value="NormalPCA">基于法向量的PCA</Option>
                      <Option value="PCA" disabled>PCA（未实现）</Option>
                      <Option value="RobPCA" disabled>鲁棒PCA（未实现）</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
            </div>

            {/* 通用参数设置 */}
            <div style={{ 
              background: '#fafafa', 
              padding: '12px 16px', 
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>通用参数设置</div>
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

            {/* 点云采集参数设置 */}
            <div style={{ 
              background: '#fafafa', 
              padding: '12px 16px', 
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>点云采集参数设置</div>
              <Row gutter={24}>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="轴朝向" 
                    name="axisOrientation"
                    tooltip="采集点云时圆柱轴线的朝向"
                  >
                    <Select>
                      <Option value="x">X轴</Option>
                      <Option value="y">Y轴</Option>
                      <Option value="z">Z轴</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="实际速度" 
                    name="actualSpeed"
                    tooltip="实际测量时的运动速度"
                  >
                    <InputNumber 
                      min={0} 
                      max={1000} 
                      style={{ width: '100%' }}
                      addonAfter="mm/s" 
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="采集速度" 
                    name="acquisitionSpeed"
                    tooltip="点云采集时的扫描速度"
                  >
                    <InputNumber 
                      min={0} 
                      max={1000} 
                      style={{ width: '100%' }}
                      addonAfter="mm/s" 
                    />
                  </Form.Item>
                </Col>
              </Row>
            </div>

            {/* RANSAC特有参数设置 */}
            {selectedMethod === 'NormalRANSAC' && (
              <div style={{ 
                background: '#fafafa', 
                padding: '12px 16px', 
                borderRadius: '4px',
                marginBottom: '16px'
              }}>
                <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>RANSAC参数设置</div>
                <Row gutter={24}>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="距离阈值" 
                      name="ransacThreshold"
                      tooltip="RANSAC算法的距离阈值"
                    >
                      <InputNumber 
                        min={0.001} 
                        max={1.0} 
                        step={0.001} 
                        style={{ width: '100%' }} 
                      />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="最大迭代次数" 
                      name="maxIterations"
                      tooltip="RANSAC算法的最大迭代次数"
                    >
                      <InputNumber 
                        min={100} 
                        max={10000} 
                        step={100}
                        style={{ width: '100%' }}
                        addonAfter="次" 
                      />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="法向量距离权重" 
                      name="normalDistanceWeight"
                      tooltip="RANSAC算法中法向量距离的权重因子"
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
              </div>
            )}

            <div>
              <Button type="primary" onClick={handlePreprocess}>
                测试轴线拟合
              </Button>
            </div>
          </Form>
        </TabPane>

        <TabPane tab="圆弧拟合" key="arc">
          <Form
            form={arcForm}
            layout="vertical"
            initialValues={{
              arcMethod: 'HyperFit',
              arcNormalNeighbors: 20,
              arcMaxRadius: 12,
              arcMinRadius: 6,
              learningRate: 0.01,
              gradientMaxIterations: 1000,
              tolerance: 1e-6,
            }}
          >
            <div style={{ 
              background: '#fafafa', 
              padding: '12px 16px', 
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>圆弧拟合方法设置</div>
              <Row gutter={24}>
                <Col span={24}>
                  <Form.Item 
                    {...formItemLayout}
                    label="圆弧拟合方法" 
                    name="arcMethod"
                    tooltip="用于拟合圆弧的算法方法"
                  >
                    <Select onChange={handleArcMethodChange}>
                      <Option value="HyperFit">超拟合算法 (Hyper Fit)</Option>
                      <Option value="PrattFit">Pratt 拟合算法</Option>
                      <Option value="TaubinFit">Taubin 拟合算法</Option>
                      <Option value="GradientDescent">Adam 优化的梯度下降法</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
            </div>

            {/* 通用圆弧拟合参数设置 */}
            <div style={{ 
              background: '#fafafa', 
              padding: '12px 16px', 
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>通用参数设置</div>
              <Row gutter={24}>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="求法线邻近点数量" 
                    name="arcNormalNeighbors"
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
                    label="最小半径" 
                    name="arcMinRadius"
                    tooltip="圆弧的最小半径限制"
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
                    name="arcMaxRadius"
                    tooltip="圆弧的最大半径限制"
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

            {/* 圆拟合迭代参数设置 */}
            <div style={{ 
              background: '#fafafa', 
              padding: '12px 16px', 
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>圆拟合迭代参数设置</div>
              <Row gutter={24}>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="拟合迭代次数" 
                    name="fitIterations"
                    tooltip="圆拟合的迭代次数"
                  >
                    <InputNumber 
                      min={1} 
                      max={1000} 
                      style={{ width: '100%' }}
                      addonAfter="次" 
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item 
                    {...formItemLayout}
                    label="采样百分比" 
                    name="samplePercentage"
                    tooltip="每次迭代采样的点数百分比"
                  >
                    <InputNumber 
                      min={1} 
                      max={100} 
                      style={{ width: '100%' }}
                      addonAfter="%" 
                    />
                  </Form.Item>
                </Col>
              </Row>
            </div>

            {/* 梯度下降法特有参数设置 */}
            {selectedArcMethod === 'GradientDescent' && (
              <div style={{ 
                background: '#fafafa', 
                padding: '12px 16px', 
                borderRadius: '4px',
                marginBottom: '16px'
              }}>
                <div style={{ marginBottom: '12px', color: '#1890ff', fontSize: '13px' }}>梯度下降法参数设置</div>
                <Row gutter={24}>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="学习率" 
                      name="learningRate"
                      tooltip="Adam优化器的学习率"
                    >
                      <InputNumber 
                        min={0.0001} 
                        max={0.1} 
                        step={0.001} 
                        style={{ width: '100%' }} 
                      />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="最大迭代次数" 
                      name="gradientMaxIterations"
                      tooltip="梯度下降的最大迭代次数"
                    >
                      <InputNumber 
                        min={100} 
                        max={10000} 
                        step={100}
                        style={{ width: '100%' }}
                        addonAfter="次" 
                      />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item 
                      {...formItemLayout}
                      label="收敛阈值" 
                      name="tolerance"
                      tooltip="梯度下降的收敛判断阈值"
                    >
                      <InputNumber 
                        min={1e-8} 
                        max={1e-4} 
                        step={1e-7}
                        style={{ width: '100%' }} 
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </div>
            )}

            <div>
              <Button type="primary" onClick={handleArcFit}>
                测试圆弧拟合
              </Button>
            </div>
          </Form>
        </TabPane>
      </Tabs>
      <div style={{ marginTop: '20px', textAlign: 'center', display: 'flex', justifyContent: 'center', gap: '16px' }}>
        <Button type="primary" size="large" onClick={handleSequentialFit}>
          执行序列拟合（轴线+圆弧）
        </Button>
        <Button size="large" onClick={handleSaveSetting}>
          保存当前设置
        </Button>
      </div>

      {/* 已保存设置列表 */}
      {savedSettings.length > 0 && (
        <Card
          className="settings-card"
          style={{ marginTop: '20px' }}
        >
          <Collapse
            defaultActiveKey={['1']}
            bordered={false}
            expandIcon={({ isActive }) => (
              <CaretRightOutlined rotate={isActive ? 90 : 0} />
            )}
          >
            <Collapse.Panel 
              header={
                <div style={{ 
                  fontSize: '16px',
                  fontWeight: 500,
                  color: '#1890ff',
                  display: 'flex',
                  alignItems: 'center'
                }}>
                  <UnorderedListOutlined style={{ marginRight: 8 }} />
                  已保存的设置 ({savedSettings.length})
                </div>
              }
              key="1"
            >
              <List
                itemLayout="horizontal"
                dataSource={savedSettings}
                renderItem={item => (
                  <List.Item
                    className="setting-list-item"
                    actions={[
                      <Button type="link" onClick={() => loadSetting(item.id)}>
                        加载
                      </Button>,
                      <Popconfirm
                        title="确定要删除这个设置吗？"
                        onConfirm={() => handleDelete(item)}
                        okText="确定"
                        cancelText="取消"
                      >
                        <Button type="text" danger icon={<DeleteOutlined />} />
                      </Popconfirm>
                    ]}
                  >
                    <List.Item.Meta
                      title={
                        <Typography.Text
                          editable={{
                            onChange: (newName) => handleUpdateName(item.id, newName),
                            tooltip: '点击修改名称'
                          }}
                          style={{ fontSize: '14px' }}
                        >
                          {item.name}
                        </Typography.Text>
                      }
                      description={
                        <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                          创建时间: {new Date(item.createdAt).toLocaleString()}
                        </Typography.Text>
                      }
                    />
                  </List.Item>
                )}
              />
            </Collapse.Panel>
          </Collapse>
        </Card>
      )}

      <style jsx="true">{`
        .settings-card {
          box-shadow: 0 1px 2px rgba(0,0,0,0.03), 
                      0 1px 6px -1px rgba(0,0,0,0.02), 
                      0 2px 4px rgba(0,0,0,0.02);
          border-radius: 8px;
        }
        
        .settings-card .ant-collapse {
          background: transparent;
        }
        
        .settings-card .ant-collapse-header {
          padding: 12px 16px !important;
        }
        
        .settings-card .setting-list-item {
          padding: 12px 24px;
          transition: background-color 0.3s;
        }
        
        .settings-card .setting-list-item:hover {
          background-color: rgba(0,0,0,0.02);
        }
        
        .settings-card .ant-typography-edit-content {
          margin-top: 0;
          margin-bottom: 0;
        }
      `}</style>

    </Card>
  );
};

export default ParamsSettingComponent;
