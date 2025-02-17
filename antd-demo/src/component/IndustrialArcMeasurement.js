import React, { useEffect, useRef, useState } from 'react';
import {
  Layout,
  Menu,
  Card,
  Typography,
  Button,
  Form,
  InputNumber,
  Upload,
  Row,
  Col,
  List,
  message,
} from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import FilterSection from './FilterSection';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

const IndustrialArcMeasurement = () => {
  // 用于切换左侧菜单的不同功能模块
  const [selectedKey, setSelectedKey] = useState('upload');

  return (
    <Layout style={{ height: '100vh', background: '#f0f2f5' }}>
      <Header style={{ background: '#001529', color: '#fff', textAlign: 'center', fontSize: 22 }}>
        圆弧测量系统
      </Header>
      <Layout>
        <Sider width={220} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={({ key }) => setSelectedKey(key)}
            style={{ height: '100%', borderRight: 0 }}
          >
            <Menu.Item key="upload">点云上传</Menu.Item>
            <Menu.Item key="filter">点云预处理</Menu.Item>
            <Menu.Item key="params">参数设置</Menu.Item>
            <Menu.Item key="display">点云显示</Menu.Item>
            <Menu.Item key="history">历史记录</Menu.Item>
          </Menu>
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 480 }}>
            {selectedKey === 'upload' && <UploadSection />}
            {selectedKey === 'filter' && <FilterSection />}
            {selectedKey === 'params' && <ParamsSection />}
            {selectedKey === 'display' && <DisplaySection />}
            {selectedKey === 'history' && <HistorySection />}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

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
          name="file"
          action="http://localhost:9304/upload"
          accept=".ply"
          multiple={false}
          showUploadList={false}
          beforeUpload={(file) => {
            const isPLY = file.name.endsWith('.ply');
            if (!isPLY) {
              message.error('只能上传PLY格式的点云文件！');
              return false;
            }
            return true;
          }}
          onChange={(info) => {
            if (info.file.status === 'uploading') {
              setUploading(true);
              setUploadStatus(null);
            } else if (info.file.status === 'done') {
              setUploading(false);
              if (info.file.response && info.file.response.error) {
                setUploadStatus('error');
                message.error(info.file.response.error);
              } else {
                setUploadStatus('success');
                message.success(`${info.file.name} 上传成功`);
                setHasUploadedCloud(true);
              }
            } else if (info.file.status === 'error') {
              setUploading(false);
              setUploadStatus('error');
              message.error(info.file.response?.error || '文件上传失败，请重试');
            }
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

const ParamsSection = () => {
  const [form] = Form.useForm();

  const handlePreprocess = () => {
    form.validateFields().then((values) => {
      // 此处可整合区域数据后提交数据
      message.success('参数提交成功！');
      console.log('提交数据:', values);
    });
  };

  return (
    <Card title="参数设置" bordered={false}>
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          downsample_rate: 5,
          ransc_downsample_rate: 5,
          nb_neighbors: 100,
          std_ratio: 0.5,
          normal_distance_weight: 0.1,
          max_iterations: 10000,
          distance_threshold: 0.1,
          radius_min: 20,
          radius_max: 55,
        }}
      >
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item label="下采样倍数" name="downsample_rate">
              <InputNumber min={1} max={10} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="RANSC 下采样倍数" name="ransc_downsample_rate">
              <InputNumber min={1} max={10} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="邻域点数" name="nb_neighbors">
              <InputNumber min={10} max={500} step={10} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="标准差比" name="std_ratio">
              <InputNumber min={0.1} max={2.0} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item label="权重" name="normal_distance_weight">
              <InputNumber min={0.01} max={1.0} step={0.01} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="最大迭代" name="max_iterations">
              <InputNumber min={100} max={50000} step={100} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="距离阈值" name="distance_threshold">
              <InputNumber min={0.01} max={1.0} step={0.01} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="最小半径" name="radius_min">
              <InputNumber min={10} max={100} step={5} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item label="最大半径" name="radius_max">
              <InputNumber min={20} max={200} step={5} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col>
            <Button type="primary" onClick={handlePreprocess}>
              点云预处理
            </Button>
          </Col>
        </Row>
      </Form>
    </Card>
  );
};

const DisplaySection = () => {
  const [showThree, setShowThree] = useState(false);
  const threeCanvasRef = useRef(null);

  useEffect(() => {
    if (showThree && threeCanvasRef.current) {
      const canvas = threeCanvasRef.current;
      const renderer = new THREE.WebGLRenderer({ canvas });
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        75,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
      );
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      camera.position.z = 5;

      const animate = () => {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };

      const loader = new PLYLoader();
      loader.load(
        '/get_ply',
        (geometry) => {
          const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.01 });
          const points = new THREE.Points(geometry, material);
          scene.add(points);
        },
        undefined,
        (error) => console.error(error)
      );
      animate();
    }
  }, [showThree]);

  return (
    <Card title="点云显示" bordered={false}>
      <Button onClick={() => setShowThree(!showThree)}>
        {showThree ? '隐藏点云' : '显示点云'}
      </Button>
      {showThree && (
        <canvas
          ref={threeCanvasRef}
          style={{
            width: '800px',
            height: '600px',
            border: '1px solid #d9d9d9',
            display: 'block',
            marginTop: 16,
          }}
        />
      )}
    </Card>
  );
};

const HistorySection = () => {
  const [history, setHistory] = useState([]);
  return (
    <Card title="历史记录" bordered={false}>
      <List
        dataSource={history}
        renderItem={(item) => <List.Item>{item}</List.Item>}
        locale={{ emptyText: '暂无历史记录' }}
      />
    </Card>
  );
};

export default IndustrialArcMeasurement;
